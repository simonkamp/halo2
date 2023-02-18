//! Chip implementations for the ECC gadgets.

use super::{BaseFitsInScalarInstructions, EccInstructions, FixedPoints};
use crate::{
    sinsemilla::primitives as sinsemilla,
    utilities::{lookup_range_check::LookupRangeCheckConfig, UtilitiesInstructions},
};
use arrayvec::ArrayVec;

use ff::{PrimeField, PrimeFieldBits};
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Error, Fixed},
};
use pasta_curves::arithmetic::CurveAffine;

use std::convert::TryInto;

pub(super) mod add;
pub(super) mod add_incomplete;
pub mod constants;
pub(super) mod mul;
pub(super) mod mul_fixed;
pub(super) mod witness_point;

pub use constants::*;

// Exposed for Sinsemilla.
pub(crate) use mul::incomplete::DoubleAndAdd;

/// A curve point represented in affine (x, y) coordinates, or the
/// identity represented as (0, 0).
/// Each coordinate is assigned to a cell.
#[derive(Clone, Debug)]
pub struct EccPoint<C: PastaCurve> {
    /// x-coordinate
    ///
    /// Stored as an `Assigned<F>` to enable batching inversions.
    x: AssignedCell<Assigned<C::Base>, C::Base>,
    /// y-coordinate
    ///
    /// Stored as an `Assigned<F>` to enable batching inversions.
    y: AssignedCell<Assigned<C::Base>, C::Base>,
}

/// The PastaCurve trait introduces some additional bounds to the CurveAffine which are needed in the ecc chip implementation.
/// TODO: This could be more cleanly expressed using the unstable associated_type_bounds feature.
pub trait PastaCurve: CurveAffine<Base = Self::PastaBase, ScalarExt = Self::PastaScalar> {
    /// bla bla
    type PastaBase: PrimeFieldBits + PrimeField<Repr = [u8; 32]> + std::cmp::PartialOrd;
    /// bla
    type PastaScalar: PrimeFieldBits + PrimeField<Repr = [u8; 32]> + std::cmp::PartialOrd;
}

impl<C> PastaCurve for C
where
    C: CurveAffine,
    C::Base: PrimeFieldBits + PrimeField<Repr = [u8; 32]>,
    C::Scalar: PrimeFieldBits + PrimeField<Repr = [u8; 32]>,
{
    type PastaBase = C::Base;
    type PastaScalar = C::Scalar;
}

impl<C: PastaCurve> EccPoint<C> {
    /// Constructs a point from its coordinates, without checking they are on the curve.
    ///
    /// This is an internal API that we only use where we know we have a valid curve point.
    pub(crate) fn from_coordinates_unchecked(
        x: AssignedCell<Assigned<C::Base>, C::Base>,
        y: AssignedCell<Assigned<C::Base>, C::Base>,
    ) -> Self {
        EccPoint { x, y }
    }

    /// Returns the value of this curve point, if known.
    pub fn point(&self) -> Value<C> {
        self.x.value().zip(self.y.value()).map(|(x, y)| {
            if x.is_zero_vartime() && y.is_zero_vartime() {
                C::identity()
            } else {
                C::from_xy(x.evaluate(), y.evaluate()).unwrap()
            }
        })
    }
    /// The cell containing the affine short-Weierstrass x-coordinate,
    /// or 0 for the zero point.
    pub fn x(&self) -> AssignedCell<C::Base, C::Base> {
        self.x.clone().evaluate()
    }
    /// The cell containing the affine short-Weierstrass y-coordinate,
    /// or 0 for the zero point.
    pub fn y(&self) -> AssignedCell<C::Base, C::Base> {
        self.y.clone().evaluate()
    }

    #[cfg(test)]
    fn is_identity(&self) -> Value<bool> {
        self.x.value().map(|x| x.is_zero_vartime())
    }
}

/// A non-identity point represented in affine (x, y) coordinates.
/// Each coordinate is assigned to a cell.
#[derive(Clone, Debug)]
pub struct NonIdentityEccPoint<C: PastaCurve> {
    /// x-coordinate
    ///
    /// Stored as an `Assigned<F>` to enable batching inversions.
    x: AssignedCell<Assigned<C::Base>, C::Base>,
    /// y-coordinate
    ///
    /// Stored as an `Assigned<F>` to enable batching inversions.
    y: AssignedCell<Assigned<C::Base>, C::Base>,
}

impl<C: PastaCurve> NonIdentityEccPoint<C> {
    /// Constructs a point from its coordinates, without checking they are on the curve.
    ///
    /// This is an internal API that we only use where we know we have a valid non-identity
    /// curve point.
    pub(crate) fn from_coordinates_unchecked(
        x: AssignedCell<Assigned<C::Base>, C::Base>,
        y: AssignedCell<Assigned<C::Base>, C::Base>,
    ) -> Self {
        NonIdentityEccPoint { x, y }
    }

    /// Returns the value of this curve point, if known.
    pub fn point(&self) -> Value<C> {
        self.x.value().zip(self.y.value()).map(|(x, y)| {
            assert!(!x.is_zero_vartime() && !y.is_zero_vartime());
            C::from_xy(x.evaluate(), y.evaluate()).unwrap()
        })
    }
    /// The cell containing the affine short-Weierstrass x-coordinate.
    pub fn x(&self) -> AssignedCell<C::Base, C::Base> {
        self.x.clone().evaluate()
    }
    /// The cell containing the affine short-Weierstrass y-coordinate.
    pub fn y(&self) -> AssignedCell<C::Base, C::Base> {
        self.y.clone().evaluate()
    }
}

impl<C: PastaCurve> From<NonIdentityEccPoint<C>> for EccPoint<C> {
    fn from(non_id_point: NonIdentityEccPoint<C>) -> Self {
        Self {
            x: non_id_point.x,
            y: non_id_point.y,
        }
    }
}

/// Configuration for [`EccChip`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(non_snake_case)]
pub struct EccConfig<C: PastaCurve, FixedPoints: super::FixedPoints<C>> {
    /// Advice columns needed by instructions in the ECC chip.
    pub advices: [Column<Advice>; 10],

    /// Incomplete addition
    add_incomplete: add_incomplete::Config<C>,

    /// Complete addition
    add: add::Config<C>,

    /// Variable-base scalar multiplication
    mul: mul::Config<C>,

    /// Fixed-base full-width scalar multiplication
    mul_fixed_full: mul_fixed::full_width::Config<C, FixedPoints>,
    /// Fixed-base signed short scalar multiplication
    mul_fixed_short: mul_fixed::short::Config<C, FixedPoints>,
    /// Fixed-base mul using a base field element as a scalar
    mul_fixed_base_field: mul_fixed::base_field_elem::Config<C, FixedPoints>,

    /// Witness point
    witness_point: witness_point::Config<C>,

    /// Lookup range check using 10-bit lookup table
    pub lookup_config: LookupRangeCheckConfig<C::Base, { sinsemilla::K }>,
}

/// A trait representing the kind of scalar used with a particular `FixedPoint`.
///
/// This trait exists because of limitations around const generics.
pub trait FixedScalarKind {
    /// The number of windows that this scalar kind requires.
    const NUM_WINDOWS: usize;
}

/// Type marker representing a full-width scalar for use in fixed-base scalar
/// multiplication.
#[derive(Debug)]
pub enum FullScalar {}
impl FixedScalarKind for FullScalar {
    const NUM_WINDOWS: usize = NUM_WINDOWS;
}

/// Type marker representing a signed 64-bit scalar for use in fixed-base scalar
/// multiplication.
#[derive(Debug)]
pub enum ShortScalar {}
impl FixedScalarKind for ShortScalar {
    const NUM_WINDOWS: usize = NUM_WINDOWS_SHORT;
}

/// Type marker representing a base field element being used as a scalar in fixed-base
/// scalar multiplication.
#[derive(Debug)]
pub enum BaseFieldElem {}
impl FixedScalarKind for BaseFieldElem {
    const NUM_WINDOWS: usize = NUM_WINDOWS;
}

/// Returns information about a fixed point that is required by [`EccChip`].
///
/// For each window required by `Self::FixedScalarKind`, $z$ is a field element such that for
/// each point $(x, y)$ in the window:
/// - $z + y = u^2$ (some square in the field); and
/// - $z - y$ is not a square.
///
/// TODO: When associated consts can be used as const generics, introduce a
/// `const NUM_WINDOWS: usize` associated const, and return `NUM_WINDOWS`-sized
/// arrays instead of `Vec`s.
pub trait FixedPoint<C: PastaCurve>: std::fmt::Debug + Eq + Clone {
    /// The kind of scalar that this fixed point can be multiplied by.
    type FixedScalarKind: FixedScalarKind;

    /// Returns the generator for this fixed point.
    fn generator(&self) -> C;

    /// Returns the $u$ values for this fixed point.
    fn u(&self) -> Vec<[<C::Base as PrimeField>::Repr; H]>;

    /// Returns the $z$ value for this fixed point.
    fn z(&self) -> Vec<u64>;

    /// Returns the Lagrange coefficients for this fixed point.
    fn lagrange_coeffs(&self) -> Vec<[C::Base; H]> {
        compute_lagrange_coeffs(self.generator(), Self::FixedScalarKind::NUM_WINDOWS)
    }
}

/// An [`EccInstructions`] chip that uses 10 advice columns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EccChip<C: PastaCurve, FixedPoints: super::FixedPoints<C>> {
    config: EccConfig<C, FixedPoints>,
}

impl<C: PastaCurve, FixedPoints: super::FixedPoints<C>> Chip<C::Base> for EccChip<C, FixedPoints> {
    type Config = EccConfig<C, FixedPoints>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<C: PastaCurve, Fixed: super::FixedPoints<C>> UtilitiesInstructions<C::Base>
    for EccChip<C, Fixed>
{
    type Var = AssignedCell<C::Base, C::Base>;
}

impl<C: PastaCurve, FixedPoints: super::FixedPoints<C>> EccChip<C, FixedPoints> {
    /// Reconstructs this chip from the given config.
    pub fn construct(config: <Self as Chip<C::Base>>::Config) -> Self {
        Self { config }
    }

    /// # Side effects
    ///
    /// All columns in `advices` will be equality-enabled.
    #[allow(non_snake_case)]
    pub fn configure(
        meta: &mut ConstraintSystem<C::Base>,
        advices: [Column<Advice>; 10],
        lagrange_coeffs: [Column<Fixed>; 8],
        range_check: LookupRangeCheckConfig<C::Base, { sinsemilla::K }>,
    ) -> <Self as Chip<C::Base>>::Config {
        // Create witness point gate
        let witness_point = witness_point::Config::configure(meta, advices[0], advices[1]);
        // Create incomplete point addition gate
        let add_incomplete =
            add_incomplete::Config::configure(meta, advices[0], advices[1], advices[2], advices[3]);

        // Create complete point addition gate
        let add = add::Config::configure(
            meta, advices[0], advices[1], advices[2], advices[3], advices[4], advices[5],
            advices[6], advices[7], advices[8],
        );

        // Create variable-base scalar mul gates
        let mul = mul::Config::configure(meta, add, range_check, advices);

        // Create config that is shared across short, base-field, and full-width
        // fixed-base scalar mul.
        let mul_fixed = mul_fixed::Config::<C, FixedPoints>::configure(
            meta,
            lagrange_coeffs,
            advices[4],
            advices[5],
            add,
            add_incomplete,
        );

        // Create gate that is only used in full-width fixed-base scalar mul.
        let mul_fixed_full =
            mul_fixed::full_width::Config::<C, FixedPoints>::configure(meta, mul_fixed.clone());

        // Create gate that is only used in short fixed-base scalar mul.
        let mul_fixed_short =
            mul_fixed::short::Config::<C, FixedPoints>::configure(meta, mul_fixed.clone());

        // Create gate that is only used in fixed-base mul using a base field element.
        let mul_fixed_base_field = mul_fixed::base_field_elem::Config::<C, FixedPoints>::configure(
            meta,
            advices[6..9].try_into().unwrap(),
            range_check,
            mul_fixed,
        );

        EccConfig {
            advices,
            add_incomplete,
            add,
            mul,
            mul_fixed_full,
            mul_fixed_short,
            mul_fixed_base_field,
            witness_point,
            lookup_config: range_check,
        }
    }
}

/// A full-width scalar used for fixed-base scalar multiplication.
/// This is decomposed into 85 3-bit windows in little-endian order,
/// i.e. `windows` = [k_0, k_1, ..., k_84] (for a 255-bit scalar)
/// where `scalar = k_0 + k_1 * (2^3) + ... + k_84 * (2^3)^84` and
/// each `k_i` is in the range [0..2^3).
#[derive(Clone, Debug)]
pub struct EccScalarFixed<C: PastaCurve> {
    value: Value<C::Scalar>,
    /// The circuit-assigned windows representing this scalar, or `None` if the scalar has
    /// not been used yet.
    windows: Option<ArrayVec<AssignedCell<C::Base, C::Base>, { NUM_WINDOWS }>>,
}

// TODO: Make V a `u64`
type MagnitudeCell<C> = AssignedCell<<C as CurveAffine>::Base, <C as CurveAffine>::Base>;
// TODO: Make V an enum Sign { Positive, Negative }
type SignCell<C> = AssignedCell<<C as CurveAffine>::Base, <C as CurveAffine>::Base>;
type MagnitudeSign<C> = (MagnitudeCell<C>, SignCell<C>);

/// A signed short scalar used for fixed-base scalar multiplication.
/// A short scalar must have magnitude in the range [0..2^64), with
/// a sign of either 1 or -1.
/// This is decomposed into 3-bit windows in little-endian order
/// using a running sum `z`, where z_{i+1} = (z_i - a_i) / (2^3)
/// for element α = a_0 + (2^3) a_1 + ... + (2^{3(n-1)}) a_{n-1}.
/// Each `a_i` is in the range [0..2^3).
///
/// `windows` = [k_0, k_1, ..., k_21] (for a 64-bit magnitude)
/// where `scalar = k_0 + k_1 * (2^3) + ... + k_84 * (2^3)^84` and
/// each `k_i` is in the range [0..2^3).
/// k_21 must be a single bit, i.e. 0 or 1.
#[derive(Clone, Debug)]
pub struct EccScalarFixedShort<C: PastaCurve> {
    magnitude: MagnitudeCell<C>,
    sign: SignCell<C>,
    /// The circuit-assigned running sum constraining this signed short scalar, or `None`
    /// if the scalar has not been used yet.
    running_sum: Option<ArrayVec<AssignedCell<C::Base, C::Base>, { NUM_WINDOWS_SHORT + 1 }>>,
}

/// A base field element used for fixed-base scalar multiplication.
/// This is decomposed into 3-bit windows in little-endian order
/// using a running sum `z`, where z_{i+1} = (z_i - a_i) / (2^3)
/// for element α = a_0 + (2^3) a_1 + ... + (2^{3(n-1)}) a_{n-1}.
/// Each `a_i` is in the range [0..2^3).
///
/// `running_sum` = [z_0, ..., z_85], where we expect z_85 = 0.
/// Since z_0 is initialized as the scalar α, we store it as
/// `base_field_elem`.
#[derive(Clone, Debug)]
struct EccBaseFieldElemFixed<C: PastaCurve> {
    base_field_elem: AssignedCell<C::Base, C::Base>,
    running_sum: ArrayVec<AssignedCell<C::Base, C::Base>, { NUM_WINDOWS + 1 }>,
}

impl<C: PastaCurve> EccBaseFieldElemFixed<C> {
    #![allow(dead_code)]
    fn base_field_elem(&self) -> AssignedCell<C::Base, C::Base> {
        self.base_field_elem.clone()
    }
}

/// An enumeration of the possible types of scalars used in variable-base
/// multiplication.
#[derive(Clone, Debug)]
pub enum ScalarVar<C: PastaCurve> {
    // todo check semantics of changing to vesta! Maybe it has to stay pallas?
    /// An element of the elliptic curve's base field, that is used as a scalar
    /// in variable-base scalar mul.
    ///
    /// It is not true in general that a scalar field element fits in a curve's
    /// base field, and in particular it is untrue for the Pallas curve, whose
    /// scalar field `Fq` is larger than its base field `Fp`.
    ///
    /// However, the only use of variable-base scalar mul in the Orchard protocol
    /// is in deriving diversified addresses `[ivk] g_d`,  and `ivk` is guaranteed
    /// to be in the base field of the curve. (See non-normative notes in
    /// [4.2.3 Orchard Key Components][orchardkeycomponents].)
    ///
    /// [orchardkeycomponents]: https://zips.z.cash/protocol/protocol.pdf#orchardkeycomponents
    BaseFieldElem(AssignedCell<C::Base, C::Base>),
    /// A full-width scalar. This is unimplemented for halo2_gadgets v0.1.0.
    FullWidth,
}

impl<C: PastaCurve, Fixed: FixedPoints<C>> EccInstructions<C> for EccChip<C, Fixed>
where
    <Fixed as FixedPoints<C>>::Base: FixedPoint<C, FixedScalarKind = BaseFieldElem>,
    <Fixed as FixedPoints<C>>::FullScalar: FixedPoint<C, FixedScalarKind = FullScalar>,
    <Fixed as FixedPoints<C>>::ShortScalar: FixedPoint<C, FixedScalarKind = ShortScalar>,
{
    type ScalarFixed = EccScalarFixed<C>;
    type ScalarFixedShort = EccScalarFixedShort<C>;
    type ScalarVar = ScalarVar<C>;
    type Point = EccPoint<C>;
    type NonIdentityPoint = NonIdentityEccPoint<C>;
    type X = AssignedCell<C::Base, C::Base>;
    type FixedPoints = Fixed;

    fn constrain_equal(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        a: &Self::Point,
        b: &Self::Point,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "constrain equal",
            |mut region| {
                // Constrain x-coordinates
                region.constrain_equal(a.x().cell(), b.x().cell())?;
                // Constrain x-coordinates
                region.constrain_equal(a.y().cell(), b.y().cell())
            },
        )
    }

    fn witness_point(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        value: Value<C>,
    ) -> Result<Self::Point, Error> {
        let config = self.config().witness_point;
        layouter.assign_region(
            || "witness point",
            |mut region| config.point(value, 0, &mut region),
        )
    }

    fn witness_point_non_id(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        value: Value<C>,
    ) -> Result<Self::NonIdentityPoint, Error> {
        let config = self.config().witness_point;
        layouter.assign_region(
            || "witness non-identity point",
            |mut region| config.point_non_id(value, 0, &mut region),
        )
    }

    fn witness_scalar_var(
        &self,
        _layouter: &mut impl Layouter<C::Base>,
        _value: Value<C::Scalar>,
    ) -> Result<Self::ScalarVar, Error> {
        // This is unimplemented for halo2_gadgets v0.1.0.
        todo!()
    }

    fn witness_scalar_fixed(
        &self,
        _layouter: &mut impl Layouter<C::Base>,
        value: Value<C::Scalar>,
    ) -> Result<Self::ScalarFixed, Error> {
        Ok(EccScalarFixed {
            value,
            // This chip uses lazy witnessing.
            windows: None,
        })
    }

    fn scalar_fixed_from_signed_short(
        &self,
        _layouter: &mut impl Layouter<C::Base>,
        (magnitude, sign): MagnitudeSign<C>,
    ) -> Result<Self::ScalarFixedShort, Error> {
        Ok(EccScalarFixedShort {
            magnitude,
            sign,
            // This chip uses lazy constraining.
            running_sum: None,
        })
    }

    fn extract_p<Point: Into<Self::Point> + Clone>(point: &Point) -> Self::X {
        let point: EccPoint<C> = (point.clone()).into();
        point.x()
    }

    fn add_incomplete(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        a: &Self::NonIdentityPoint,
        b: &Self::NonIdentityPoint,
    ) -> Result<Self::NonIdentityPoint, Error> {
        let config = self.config().add_incomplete;
        layouter.assign_region(
            || "incomplete point addition",
            |mut region| config.assign_region(a, b, 0, &mut region),
        )
    }

    fn add<A: Into<Self::Point> + Clone, B: Into<Self::Point> + Clone>(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        a: &A,
        b: &B,
    ) -> Result<Self::Point, Error> {
        let config = self.config().add;
        layouter.assign_region(
            || "complete point addition",
            |mut region| {
                config.assign_region(&(a.clone()).into(), &(b.clone()).into(), 0, &mut region)
            },
        )
    }

    fn mul(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        scalar: &Self::ScalarVar,
        base: &Self::NonIdentityPoint,
    ) -> Result<(Self::Point, Self::ScalarVar), Error> {
        let config = self.config().mul;
        match scalar {
            ScalarVar::BaseFieldElem(scalar) => config.assign(
                layouter.namespace(|| "variable-base scalar mul"),
                scalar.clone(),
                base,
            ),
            ScalarVar::FullWidth => {
                todo!()
            }
        }
    }

    fn mul_fixed(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        scalar: &Self::ScalarFixed,
        base: &<Self::FixedPoints as FixedPoints<C>>::FullScalar,
    ) -> Result<(Self::Point, Self::ScalarFixed), Error> {
        let config = self.config().mul_fixed_full.clone();
        config.assign(
            layouter.namespace(|| format!("fixed-base mul of {:?}", base)),
            scalar,
            base,
        )
    }

    fn mul_fixed_short(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        scalar: &Self::ScalarFixedShort,
        base: &<Self::FixedPoints as FixedPoints<C>>::ShortScalar,
    ) -> Result<(Self::Point, Self::ScalarFixedShort), Error> {
        let config = self.config().mul_fixed_short.clone();
        config.assign(
            layouter.namespace(|| format!("short fixed-base mul of {:?}", base)),
            scalar,
            base,
        )
    }

    fn mul_fixed_base_field_elem(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        base_field_elem: AssignedCell<C::Base, C::Base>,
        base: &<Self::FixedPoints as FixedPoints<C>>::Base,
    ) -> Result<Self::Point, Error> {
        let config = self.config().mul_fixed_base_field.clone();
        config.assign(
            layouter.namespace(|| format!("base-field elem fixed-base mul of {:?}", base)),
            base_field_elem,
            base,
        )
    }
}

impl<C: PastaCurve, Fixed: FixedPoints<C>> BaseFitsInScalarInstructions<C> for EccChip<C, Fixed>
where
    <Fixed as FixedPoints<C>>::Base: FixedPoint<C, FixedScalarKind = BaseFieldElem>,
    <Fixed as FixedPoints<C>>::FullScalar: FixedPoint<C, FixedScalarKind = FullScalar>,
    <Fixed as FixedPoints<C>>::ShortScalar: FixedPoint<C, FixedScalarKind = ShortScalar>,
{
    fn scalar_var_from_base(
        &self,
        _layouter: &mut impl Layouter<C::Base>,
        base: &Self::Var,
    ) -> Result<Self::ScalarVar, Error> {
        Ok(ScalarVar::BaseFieldElem(base.clone()))
    }
}
