use std::ops::Deref;

use pasta_curves::arithmetic::CurveAffine;

use crate::{
    plonk::{ChallengeTheta, Expression},
    poly::{self, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial},
};
use ff::Field;

#[derive(Debug)]
pub(crate) struct Context<'a, C: CurveAffine, Ev, Ec>
where
    Ev: Copy + Send + Sync,
    Ec: Copy + Send + Sync,
{
    pub(crate) domain: &'a EvaluationDomain<C::Scalar>,
    pub(crate) value_evaluator: &'a poly::Evaluator<Ev, C::Scalar, LagrangeCoeff>,
    pub(crate) advice_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
    pub(crate) fixed_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
    pub(crate) instance_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
    pub(crate) advice_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
    pub(crate) fixed_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
    pub(crate) instance_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
}

#[derive(Clone, Debug)]
pub(crate) struct UncompressedExpressions<F>(pub Vec<Expression<F>>);

impl<F> From<Vec<Expression<F>>> for UncompressedExpressions<F> {
    fn from(expressions: Vec<Expression<F>>) -> Self {
        Self(expressions)
    }
}

impl<F: Field> Deref for UncompressedExpressions<F> {
    type Target = Vec<Expression<F>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: ff::WithSmallOrderMulGroup<3>> UncompressedExpressions<F> {
    pub(in crate::plonk) fn compress<'a, C: CurveAffine, Ev, Ec>(
        &self,
        theta: ChallengeTheta<C>,
        context: &'a Context<'a, C, Ev, Ec>,
    ) -> (
        poly::Ast<Ec, C::Scalar, ExtendedLagrangeCoeff>,
        Polynomial<C::Scalar, LagrangeCoeff>,
    )
    where
        Ev: Copy + Send + Sync,
        Ec: Copy + Send + Sync,
        C: CurveAffine<ScalarExt = F>,
    {
        // Values of expressions
        let expression_values: Vec<_> = self
            .iter()
            .map(|expression| {
                expression.evaluate(
                    &|scalar| poly::Ast::ConstantTerm(scalar),
                    &|_| panic!("virtual selectors are removed during optimization"),
                    &|query| {
                        context.fixed_values[query.column_index]
                            .with_rotation(query.rotation)
                            .into()
                    },
                    &|query| {
                        context.advice_values[query.column_index]
                            .with_rotation(query.rotation)
                            .into()
                    },
                    &|query| {
                        context.instance_values[query.column_index]
                            .with_rotation(query.rotation)
                            .into()
                    },
                    &|a| -a,
                    &|a, b| a + b,
                    &|a, b| a * b,
                    &|a, scalar| a * scalar,
                )
            })
            .collect();

        let cosets: Vec<_> = self
            .iter()
            .map(|expression| {
                expression.evaluate(
                    &|scalar| poly::Ast::ConstantTerm(scalar),
                    &|_| panic!("virtual selectors are removed during optimization"),
                    &|query| {
                        context.fixed_cosets[query.column_index]
                            .with_rotation(query.rotation)
                            .into()
                    },
                    &|query| {
                        context.advice_cosets[query.column_index]
                            .with_rotation(query.rotation)
                            .into()
                    },
                    &|query| {
                        context.instance_cosets[query.column_index]
                            .with_rotation(query.rotation)
                            .into()
                    },
                    &|a| -a,
                    &|a, b| a + b,
                    &|a, b| a * b,
                    &|a, scalar| a * scalar,
                )
            })
            .collect();

        // Compressed version of expressions
        let compressed_expressions = expression_values.iter().fold(
            poly::Ast::ConstantTerm(C::Scalar::ZERO),
            |acc, expression| &(acc * *theta) + expression,
        );

        // Compressed version of cosets
        let compressed_cosets = cosets.iter().fold(
            poly::Ast::<_, _, ExtendedLagrangeCoeff>::ConstantTerm(C::Scalar::ZERO),
            |acc, eval| acc * poly::Ast::ConstantTerm(*theta) + eval.clone(),
        );

        (
            compressed_cosets,
            context
                .value_evaluator
                .evaluate(&compressed_expressions, context.domain),
        )
    }
}
