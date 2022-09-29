use std::ops::{Mul, MulAssign};

use ff::{BatchInvert, Field};
use group::Curve;
use pasta_curves::arithmetic::CurveAffine;
use rand_core::RngCore;

use super::Argument;
use crate::{
    arithmetic::{eval_polynomial, parallelize},
    plonk::{
        lookup::compression::Context, ChallengeBeta, ChallengeTheta, ChallengeX, Error, ProvingKey,
    },
    poly::{
        self,
        commitment::{Blind, Params},
        multiopen::ProverQuery,
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

#[derive(Debug)]
pub(in crate::plonk) struct Compressed<C: CurveAffine, Ev> {
    original_cosets: poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>,
    original: Polynomial<C::Scalar, LagrangeCoeff>,
    permuted_cosets: poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>,
    permuted: Polynomial<C::Scalar, LagrangeCoeff>,
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine, Ev> {
    compressed: Compressed<C, Ev>,
    product_poly: Polynomial<C::Scalar, Coeff>,
    product_coset: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    product_blind: Blind<C::Scalar>,
}

#[derive(Debug)]
pub(in crate::plonk) struct Constructed<C: CurveAffine> {
    product_poly: Polynomial<C::Scalar, Coeff>,
    product_blind: Blind<C::Scalar>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Constructed<C>,
}

impl<F: ff::WithSmallOrderMulGroup<3>> Argument<F> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn compress_expressions<
        'a,
        C,
        Ev: Copy + Send + Sync,
        Ec: Copy + Send + Sync,
    >(
        &self,
        domain: &EvaluationDomain<C::Scalar>,
        value_evaluator: &poly::Evaluator<Ev, C::Scalar, LagrangeCoeff>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
        fixed_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
        instance_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
        advice_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
        fixed_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
        instance_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
    ) -> Compressed<C, Ec>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        let context = Context {
            domain,
            value_evaluator,
            advice_values,
            fixed_values,
            instance_values,
            advice_cosets,
            fixed_cosets,
            instance_cosets,
        };

        let (original_cosets, original) = self.original_expressions.compress(theta, &context);
        let (permuted_cosets, permuted) = self.permuted_expressions.compress(theta, &context);

        Compressed {
            original_cosets,
            original,
            permuted_cosets,
            permuted,
        }
    }
}

impl<C: CurveAffine, Ev> Compressed<C, Ev> {
    pub(in crate::plonk) fn commit_product<
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        beta: ChallengeBeta<C>,
        evaluator: &mut poly::Evaluator<Ev, C::Scalar, ExtendedLagrangeCoeff>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Committed<C, Ev>, Error> {
        let blinding_factors = pk.vk.cs.blinding_factors();
        // Goal is to compute the products of fractions
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        // Denominator: (\theta^{m-1} a'_0(\omega^i) + \theta^{m-2} a'_1(\omega^i) + ... + \theta a'_{m-2}(\omega^i) + a'_{m-1}(\omega^i) + \beta)
        //
        // where a(X) is the compression of the original expressions in this multiset equality check,
        // a'(X) is the compression of the permuted expressions,
        // and i is the ith row of the expression.
        let mut product = vec![C::Scalar::ZERO; params.n as usize];

        // Denominator uses the permuted expression
        parallelize(&mut product, |product, start| {
            for (product, permuted_value) in product.iter_mut().zip(self.permuted[start..].iter()) {
                *product = *beta + permuted_value;
            }
        });

        // Batch invert to obtain the denominators for the product
        // polynomials
        product.iter_mut().batch_invert();

        // Finish the computation of the entire fraction by computing the numerators
        // (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        parallelize(&mut product, |product, start| {
            for (product, original_value) in product.iter_mut().zip(self.original[start..].iter()) {
                *product *= &(*beta + original_value);
            }
        });

        // The product vector is a vector of products of fractions of the form
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        // Denominator: (\theta^{m-1} a'_0(\omega^i) + \theta^{m-2} a'_1(\omega^i) + ... + \theta a'_{m-2}(\omega^i) + a'_{m-1}(\omega^i) + \beta)
        //
        // where there are m original expressions and m permuted expressions,
        // a_j(\omega^i)'s are the original expressions,
        // a'_j(\omega^i)'s are the original expressions,
        // and i is the ith row of the expression.

        // Compute the evaluations of the lookup product polynomial
        // over our domain, starting with z[0] = 1
        let z = std::iter::once(C::Scalar::ONE)
            .chain(product)
            .scan(C::Scalar::ONE, |state, cur| {
                *state *= &cur;
                Some(*state)
            })
            // Take all rows including the "last" row which should
            // be a boolean (and ideally 1, else soundness is broken)
            .take(params.n as usize - blinding_factors)
            // Chain random blinding factors.
            .chain((0..blinding_factors).map(|_| C::Scalar::random(&mut rng)))
            .collect::<Vec<_>>();
        assert_eq!(z.len(), params.n as usize);
        let z = pk.vk.domain.lagrange_from_vec(z);

        #[cfg(feature = "sanity-checks")]
        // This test works only with intermediate representations in this method.
        // It can be used for debugging purposes.
        {
            // While in Lagrange basis, check that product is correctly constructed
            let u = (params.n as usize) - (blinding_factors + 1);

            // l_0(X) * (1 - z(X)) = 0
            assert_eq!(z[0], C::Scalar::ONE);

            // z(\omega X) (\theta^{m-1} a'_0(X) + ... + a'_{m-1}(X) + \beta)
            // - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
            for i in 0..u {
                let mut left = z[i + 1];
                let permuted_value = &self.permuted[i];

                left *= &(*beta + permuted_value);

                let mut right = z[i];
                let original_value = self.original[i];

                right *= &(*beta + original_value);

                assert_eq!(left, right);
            }

            // l_last(X) * (z(X)^2 - z(X)) = 0
            // Assertion will fail only when soundness is broken, in which
            // case this z[u] value will be zero. (bad!)
            assert_eq!(z[u], C::Scalar::ONE);
        }

        let product_blind = Blind(C::Scalar::random(rng));
        let product_commitment = params.commit_lagrange(&z, product_blind).to_affine();
        let z = pk.vk.domain.lagrange_to_coeff(z);
        let product_coset = evaluator.register_poly(pk.vk.domain.coeff_to_extended(z.clone()));

        // Hash product commitment
        transcript.write_point(product_commitment)?;

        Ok(Committed::<C, _> {
            compressed: self,
            product_poly: z,
            product_coset,
            product_blind,
        })
    }
}

impl<'a, C: CurveAffine, Ev: Copy + Send + Sync + 'a> Committed<C, Ev> {
    /// Given a Multiset equality argument with unpermuted expressions,
    /// permuted expressions, and grand product polynomial, this method
    /// constructs constraints that must hold between these values.
    /// This method returns the constraints as a vector of ASTs for polynomials in
    /// the extended evaluation domain.
    pub(in crate::plonk) fn construct(
        self,
        beta: ChallengeBeta<C>,
        l0: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
        l_blind: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
        l_last: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    ) -> (
        Constructed<C>,
        impl Iterator<Item = poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>> + 'a,
    ) {
        let compressed = self.compressed;

        let active_rows = poly::Ast::one() - (poly::Ast::from(l_last) + l_blind);
        let beta = poly::Ast::ConstantTerm(*beta);

        let expressions = std::iter::empty()
            // l_0(X) * (1 - z(X)) = 0
            .chain(Some((poly::Ast::one() - self.product_coset) * l0))
            // l_last(X) * (z(X)^2 - z(X)) = 0
            .chain(Some(
                (poly::Ast::from(self.product_coset) * self.product_coset - self.product_coset)
                    * l_last,
            ))
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z(\omega X) (\theta^{m-1} a'_0(X) + ... + a'_{m-1}(X) + \beta)
            //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
            // ) = 0
            .chain({
                // z(\omega X) (a'(X) + \beta)
                let left: poly::Ast<_, _, _> = poly::Ast::<_, C::Scalar, _>::from(
                    self.product_coset.with_rotation(Rotation::next()),
                ) * (compressed.permuted_cosets + beta.clone());

                //  z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
                let right: poly::Ast<_, _, _> =
                    poly::Ast::from(self.product_coset) * (&compressed.original_cosets + &beta);

                Some((left - right) * active_rows)
            });

        (
            Constructed {
                product_poly: self.product_poly,
                product_blind: self.product_blind,
            },
            expressions,
        )
    }
}

impl<C: CurveAffine> Constructed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        pk: &ProvingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &pk.vk.domain;
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let product_eval = eval_polynomial(&self.product_poly, *x);
        let product_next_eval = eval_polynomial(&self.product_poly, x_next);

        // Hash each advice evaluation
        for eval in std::iter::empty()
            .chain(Some(product_eval))
            .chain(Some(product_next_eval))
        {
            transcript.write_scalar(eval)?;
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        std::iter::empty()
            // Open multiset argument product commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.product_poly,
                blind: self.constructed.product_blind,
            }))
            // Open multiset argument product commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.product_poly,
                blind: self.constructed.product_blind,
            }))
    }
}
