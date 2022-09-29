use super::super::{ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX, Error, ProvingKey};
use super::compression::Context;
use super::multiset_equality::prover::Compressed;
use super::Argument;
use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    poly::{
        self,
        commitment::{Blind, Params},
        multiopen::ProverQuery,
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::WithSmallOrderMulGroup;
use group::{ff::Field, Curve};
use rand_core::RngCore;
use std::{
    collections::BTreeMap,
    iter,
    ops::{Mul, MulAssign},
};

#[derive(Debug)]
struct PermutedInner<C: CurveAffine, Ev> {
    coset_leaf: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    poly: Polynomial<C::Scalar, Coeff>,
    blind: Blind<C::Scalar>,
}

#[derive(Debug)]
pub(in crate::plonk) struct Permuted<C: CurveAffine, Ev> {
    compressed_input: Compressed<C, Ev>,
    input: PermutedInner<C, Ev>,
    compressed_table: Compressed<C, Ev>,
    table: PermutedInner<C, Ev>,
}

#[derive(Debug)]
struct CommittedInner<C: CurveAffine, Ev> {
    permuted: PermutedInner<C, Ev>,
    product_poly: Polynomial<C::Scalar, Coeff>,
    product_coset: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    product_blind: Blind<C::Scalar>,
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine, Ev> {
    input: CommittedInner<C, Ev>,
    table: CommittedInner<C, Ev>,
}

struct ConstructedInner<C: CurveAffine> {
    permuted_poly: Polynomial<C::Scalar, Coeff>,
    permuted_blind: Blind<C::Scalar>,
    product_poly: Polynomial<C::Scalar, Coeff>,
    product_blind: Blind<C::Scalar>,
}

pub(in crate::plonk) struct Constructed<C: CurveAffine> {
    input: ConstructedInner<C>,
    table: ConstructedInner<C>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Constructed<C>,
}

impl<F: WithSmallOrderMulGroup<3>> Argument<F> {
    /// Given a Lookup with input expressions [A_0, A_1, ..., A_{m-1}] and table expressions
    /// [S_0, S_1, ..., S_{m-1}], this method
    /// - constructs A_compressed = \theta^{m-1} A_0 + theta^{m-2} A_1 + ... + \theta A_{m-2} + A_{m-1}
    ///   and S_compressed = \theta^{m-1} S_0 + theta^{m-2} S_1 + ... + \theta S_{m-2} + S_{m-1},
    /// - permutes A_compressed and S_compressed using permute_expression_pair() helper,
    ///   obtaining A' and S', and
    /// - constructs Permuted<C> struct using permuted_input_value = A', and
    ///   permuted_table_expression = S'.
    /// The Permuted<C> struct is used to update the Lookup, and is then returned.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn commit_permuted<
        'a,
        C,
        E: EncodedChallenge<C>,
        Ev: Copy + Send + Sync,
        Ec: Copy + Send + Sync,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        &self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        domain: &EvaluationDomain<C::Scalar>,
        value_evaluator: &poly::Evaluator<Ev, C::Scalar, LagrangeCoeff>,
        coset_evaluator: &mut poly::Evaluator<Ec, C::Scalar, ExtendedLagrangeCoeff>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
        fixed_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
        instance_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
        advice_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
        fixed_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
        instance_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Permuted<C, Ec>, Error>
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

        // Get values of input expressions involved in the lookup and compress them
        let (compressed_input_coset, compressed_input_expression) =
            self.input_expressions.compress(theta, &context);

        // Get values of table expressions involved in the lookup and compress them
        let (compressed_table_coset, compressed_table_expression) =
            self.table_expressions.compress(theta, &context);

        // Permute compressed (InputExpression, TableExpression) pair
        let (permuted_input_expression, permuted_table_expression) = permute_expression_pair::<C, _>(
            pk,
            params,
            domain,
            &mut rng,
            &compressed_input_expression,
            &compressed_table_expression,
        )?;

        // Closure to construct commitment to vector of values
        let mut commit_values = |values: &Polynomial<C::Scalar, LagrangeCoeff>| {
            let poly = pk.vk.domain.lagrange_to_coeff(values.clone());
            let blind = Blind(C::Scalar::random(&mut rng));
            let commitment = params.commit_lagrange(values, blind).to_affine();
            (poly, blind, commitment)
        };

        // Commit to permuted input expression
        let (permuted_input_poly, permuted_input_blind, permuted_input_commitment) =
            commit_values(&permuted_input_expression);

        // Commit to permuted table expression
        let (permuted_table_poly, permuted_table_blind, permuted_table_commitment) =
            commit_values(&permuted_table_expression);

        // Hash permuted input commitment
        transcript.write_point(permuted_input_commitment)?;

        // Hash permuted table commitment
        transcript.write_point(permuted_table_commitment)?;

        let permuted_input_coset_leaf = coset_evaluator
            .register_poly(pk.vk.domain.coeff_to_extended(permuted_input_poly.clone()));
        let permuted_table_coset_leaf = coset_evaluator
            .register_poly(pk.vk.domain.coeff_to_extended(permuted_table_poly.clone()));

        let compressed_input = Compressed {
            original_cosets: compressed_input_coset,
            original: compressed_input_expression,
            permuted_cosets: permuted_input_coset_leaf.into(),
            permuted: permuted_input_expression,
        };
        let input = PermutedInner {
            coset_leaf: permuted_input_coset_leaf,
            poly: permuted_input_poly,
            blind: permuted_input_blind,
        };

        let compressed_table = Compressed {
            original_cosets: compressed_table_coset,
            original: compressed_table_expression,
            permuted_cosets: permuted_table_coset_leaf.into(),
            permuted: permuted_table_expression,
        };
        let table = PermutedInner {
            coset_leaf: permuted_table_coset_leaf,
            poly: permuted_table_poly,
            blind: permuted_table_blind,
        };

        Ok(Permuted {
            compressed_input,
            input,
            compressed_table,
            table,
        })
    }
}

impl<C: CurveAffine, Ev: Copy + Send + Sync> Permuted<C, Ev> {
    /// Given a Lookup with input expressions, table expressions, and the permuted
    /// input expression and permuted table expression, this method constructs the
    /// grand product polynomial over the lookup. The grand product polynomial
    /// is used to populate the Product<C> struct. The Product<C> struct is
    /// added to the Lookup and finally returned by the method.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn commit_product<
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        evaluator: &mut poly::Evaluator<Ev, C::Scalar, ExtendedLagrangeCoeff>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Committed<C, Ev>, Error> {
        let input = self
            .compressed_input
            .commit_product(pk, params, beta, evaluator, &mut rng, transcript)?;
        let table = self
            .compressed_table
            .commit_product(pk, params, gamma, evaluator, rng, transcript)?;

        let input = CommittedInner {
            permuted: self.input,
            product_poly: input.product_poly,
            product_coset: input.product_coset,
            product_blind: input.product_blind,
        };

        let table = CommittedInner {
            permuted: self.table,
            product_poly: table.product_poly,
            product_coset: table.product_coset,
            product_blind: table.product_blind,
        };

        Ok(Committed::<C, _> { input, table })
    }
}

impl<'a, C: CurveAffine, Ev: Copy + Send + Sync + 'a> Committed<C, Ev> {
    /// Given a Lookup with input expressions, table expressions, permuted input
    /// expression, permuted table expression, and grand product polynomial, this
    /// method constructs constraints that must hold between these values.
    /// This method returns the constraints as a vector of ASTs for polynomials in
    /// the extended evaluation domain.
    pub(in crate::plonk) fn construct(
        self,
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        l0: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
        l_blind: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
        l_last: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    ) -> (
        Constructed<C>,
        impl Iterator<Item = poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>> + 'a,
    ) {
        let active_rows = poly::Ast::one() - (poly::Ast::from(l_last) + l_blind);
        let beta = poly::Ast::ConstantTerm(*beta);
        let gamma = poly::Ast::ConstantTerm(*gamma);

        let expressions = iter::empty()
            // l_0(X) * (1 - z_input(X)) = 0
            .chain(Some((poly::Ast::one() - self.input.product_coset) * l0))
            // l_0(X) * (1 - z_table(X)) = 0
            .chain(Some((poly::Ast::one() - self.table.product_coset) * l0))
            // l_last(X) * (z_input(X)^2 - z_input(X)) = 0
            .chain(Some(
                (poly::Ast::from(self.input.product_coset)
                    * poly::Ast::from(self.input.product_coset)
                    - poly::Ast::from(self.input.product_coset))
                    * l_last,
            ))
            // l_last(X) * (z_table(X)^2 - z_table(X)) = 0
            .chain(Some(
                (poly::Ast::from(self.table.product_coset)
                    * poly::Ast::from(self.table.product_coset)
                    - poly::Ast::from(self.table.product_coset))
                    * l_last,
            ))
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z_input(\omega X) (a'(X) + \beta)
            //   - z_input(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
            // ) = 0
            .chain({
                // z_input(\omega X) (a'(X) + \beta)
                let left: poly::Ast<_, _, _> =
                    poly::Ast::from(self.input.product_coset.with_rotation(Rotation::next()))
                        * (poly::Ast::from(self.input.permuted.coset_leaf) + beta.clone());

                //  z_input(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
                let right: poly::Ast<_, _, _> = poly::Ast::from(self.input.product_coset)
                    * (poly::Ast::from(self.input.permuted.coset_leaf) + beta);

                Some((left - right) * active_rows.clone())
            })
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z_table(\omega X) (s'(X) + \gamma)
            //   - z_table(X) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            // ) = 0
            .chain({
                // z_table(\omega X) (s'(X) + \gamma)
                let left: poly::Ast<_, _, _> =
                    poly::Ast::from(self.table.product_coset.with_rotation(Rotation::next()))
                        * (poly::Ast::from(self.table.permuted.coset_leaf) + gamma.clone());

                //  z_table(X) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
                let right: poly::Ast<_, _, _> = poly::Ast::from(self.table.product_coset)
                    * (poly::Ast::from(self.table.permuted.coset_leaf) + gamma);

                Some((left - right) * active_rows.clone())
            })
            // Check that the first values in the permuted input expression and permuted
            // fixed expression are the same.
            // l_0(X) * (a'(X) - s'(X)) = 0
            .chain(Some(
                (poly::Ast::from(self.input.permuted.coset_leaf)
                    - poly::Ast::from(self.table.permuted.coset_leaf))
                    * l0,
            ))
            // Check that each value in the permuted lookup input expression is either
            // equal to the value above it, or the value at the same index in the
            // permuted table expression.
            // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
            .chain(Some(
                (poly::Ast::from(self.input.permuted.coset_leaf)
                    - poly::Ast::from(self.table.permuted.coset_leaf))
                    * (poly::Ast::from(self.input.permuted.coset_leaf)
                        - self
                            .input
                            .permuted
                            .coset_leaf
                            .with_rotation(Rotation::prev()))
                    * active_rows,
            ));

        (
            Constructed {
                input: ConstructedInner {
                    permuted_poly: self.input.permuted.poly,
                    permuted_blind: self.input.permuted.blind,
                    product_poly: self.input.product_poly,
                    product_blind: self.input.product_blind,
                },
                table: ConstructedInner {
                    permuted_poly: self.table.permuted.poly,
                    permuted_blind: self.table.permuted.blind,
                    product_poly: self.table.product_poly,
                    product_blind: self.table.product_blind,
                },
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
        let x_inv = domain.rotate_omega(*x, Rotation::prev());
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let product_input_eval = eval_polynomial(&self.input.product_poly, *x);
        let product_input_next_eval = eval_polynomial(&self.input.product_poly, x_next);
        let product_table_eval = eval_polynomial(&self.table.product_poly, *x);
        let product_table_next_eval = eval_polynomial(&self.table.product_poly, x_next);
        let permuted_input_eval = eval_polynomial(&self.input.permuted_poly, *x);
        let permuted_input_inv_eval = eval_polynomial(&self.input.permuted_poly, x_inv);
        let permuted_table_eval = eval_polynomial(&self.table.permuted_poly, *x);

        // Hash each advice evaluation
        for eval in iter::empty()
            .chain(Some(product_input_eval))
            .chain(Some(product_input_next_eval))
            .chain(Some(product_table_eval))
            .chain(Some(product_table_next_eval))
            .chain(Some(permuted_input_eval))
            .chain(Some(permuted_input_inv_eval))
            .chain(Some(permuted_table_eval))
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
        let x_inv = pk.vk.domain.rotate_omega(*x, Rotation::prev());
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            // Open lookup input product commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.input.product_poly,
                blind: self.constructed.input.product_blind,
            }))
            // Open lookup table product commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.table.product_poly,
                blind: self.constructed.table.product_blind,
            }))
            // Open lookup input commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.input.permuted_poly,
                blind: self.constructed.input.permuted_blind,
            }))
            // Open lookup table commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.table.permuted_poly,
                blind: self.constructed.table.permuted_blind,
            }))
            // Open lookup input commitments at x_inv
            .chain(Some(ProverQuery {
                point: x_inv,
                poly: &self.constructed.input.permuted_poly,
                blind: self.constructed.input.permuted_blind,
            }))
            // Open lookup input product commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.input.product_poly,
                blind: self.constructed.input.product_blind,
            }))
            // Open lookup table product commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.table.product_poly,
                blind: self.constructed.table.product_blind,
            }))
    }
}

type ExpressionPair<F> = (Polynomial<F, LagrangeCoeff>, Polynomial<F, LagrangeCoeff>);

/// Given a vector of input values A and a vector of table values S,
/// this method permutes A and S to produce A' and S', such that:
/// - like values in A' are vertically adjacent to each other; and
/// - the first row in a sequence of like values in A' is the row
///   that has the corresponding value in S'.
/// This method returns (A', S') if no errors are encountered.
fn permute_expression_pair<C: CurveAffine, R: RngCore>(
    pk: &ProvingKey<C>,
    params: &Params<C>,
    domain: &EvaluationDomain<C::Scalar>,
    mut rng: R,
    input_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    table_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
) -> Result<ExpressionPair<C::Scalar>, Error> {
    let blinding_factors = pk.vk.cs.blinding_factors();
    let usable_rows = params.n as usize - (blinding_factors + 1);

    let mut permuted_input_expression: Vec<C::Scalar> = input_expression.to_vec();
    permuted_input_expression.truncate(usable_rows);

    // Sort input lookup expression values
    permuted_input_expression.sort();

    // A BTreeMap of each unique element in the table expression and its count
    let mut leftover_table_map: BTreeMap<C::Scalar, u32> = table_expression
        .iter()
        .take(usable_rows)
        .fold(BTreeMap::new(), |mut acc, coeff| {
            *acc.entry(*coeff).or_insert(0) += 1;
            acc
        });
    let mut permuted_table_coeffs = vec![C::Scalar::ZERO; usable_rows];

    let mut repeated_input_rows = permuted_input_expression
        .iter()
        .zip(permuted_table_coeffs.iter_mut())
        .enumerate()
        .filter_map(|(row, (input_value, table_value))| {
            // If this is the first occurrence of `input_value` in the input expression
            if row == 0 || *input_value != permuted_input_expression[row - 1] {
                *table_value = *input_value;
                // Remove one instance of input_value from leftover_table_map
                if let Some(count) = leftover_table_map.get_mut(input_value) {
                    assert!(*count > 0);
                    *count -= 1;
                    None
                } else {
                    // Return error if input_value not found
                    Some(Err(Error::ConstraintSystemFailure))
                }
            // If input value is repeated
            } else {
                Some(Ok(row))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Populate permuted table at unfilled rows with leftover table elements
    for (coeff, count) in leftover_table_map.iter() {
        for _ in 0..*count {
            permuted_table_coeffs[repeated_input_rows.pop().unwrap()] = *coeff;
        }
    }
    assert!(repeated_input_rows.is_empty());

    permuted_input_expression
        .extend((0..(blinding_factors + 1)).map(|_| C::Scalar::random(&mut rng)));
    permuted_table_coeffs.extend((0..(blinding_factors + 1)).map(|_| C::Scalar::random(&mut rng)));
    assert_eq!(permuted_input_expression.len(), params.n as usize);
    assert_eq!(permuted_table_coeffs.len(), params.n as usize);

    #[cfg(feature = "sanity-checks")]
    {
        let mut last = None;
        for (a, b) in permuted_input_expression
            .iter()
            .zip(permuted_table_coeffs.iter())
            .take(usable_rows)
        {
            if *a != *b {
                assert_eq!(*a, last.unwrap());
            }
            last = Some(*a);
        }
    }

    Ok((
        domain.lagrange_from_vec(permuted_input_expression),
        domain.lagrange_from_vec(permuted_table_coeffs),
    ))
}
