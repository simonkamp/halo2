use ff::Field;

use super::{compression::UncompressedExpressions, Expression};

pub(crate) mod prover;
pub(crate) mod verifier;

#[derive(Clone, Debug)]
pub(crate) struct Argument<F: Field> {
    pub original_expressions: UncompressedExpressions<F>,
    pub permuted_expressions: UncompressedExpressions<F>,
}

impl<F: Field> Argument<F> {
    /// Constructs a new multiset equality argument.
    ///
    /// `multiset_map` is a sequence of `(original, permuted)` tuples.
    pub fn new(multiset_map: Vec<(Expression<F>, Expression<F>)>) -> Self {
        let (original_expressions, permuted_expressions): (Vec<Expression<F>>, Vec<Expression<F>>) =
            multiset_map.into_iter().unzip();
        Argument {
            original_expressions: original_expressions.into(),
            permuted_expressions: permuted_expressions.into(),
        }
    }

    pub(crate) fn required_degree(&self) -> usize {
        assert_eq!(
            self.original_expressions.0.len(),
            self.permuted_expressions.0.len()
        );

        // The first value in the permutation poly should be one.
        // degree 2:
        // l_0(X) * (1 - z(X)) = 0
        //
        // The "last" value in the permutation poly should be a boolean, for
        // completeness and soundness.
        // degree 3:
        // l_last(X) * (z(X)^2 - z(X)) = 0
        //
        // Enable the permutation argument for only the rows involved.
        // degree (2 + original_degree) or (2 + permuted_degree) or 3,
        // whichever is larger:
        // (1 - (l_last(X) + l_blind(X))) * (
        //   z(\omega X) (\theta^{m-1} a'_0(X) + ... + a'_{m-1}(X) + \beta)
        //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
        // ) = 0
        //
        let mut original_degree = 1;
        for expr in self.original_expressions.0.iter() {
            original_degree = std::cmp::max(original_degree, expr.degree());
        }
        let mut permuted_degree = 1;
        for expr in self.permuted_expressions.0.iter() {
            permuted_degree = std::cmp::max(permuted_degree, expr.degree());
        }

        // In practice because original_degree and permuted_degree are initialized to
        // one, the latter half of this max() invocation is at least 3 always,
        // rendering this call pointless except to be explicit in case we change
        // the initialization of original_degree/permuted_degree in the future.
        std::cmp::max(
            // (1 - (l_last + l_blind)) z(\omega X) (\theta^{m-1} a'_0(X) + ... + a'_{m-1}(X) + \beta)
            2 + original_degree,
            // (1 - (l_last + l_blind)) z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
            2 + original_degree,
        )
    }
}
