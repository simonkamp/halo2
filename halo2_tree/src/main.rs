mod pallas_zs_and_us;
mod vesta_zs_and_us;

use std::{fmt::Debug, marker::PhantomData};

use halo2_gadgets::{
    ecc::{
        chip::{
            BaseFieldElem, EccChip, EccConfig, FixedPoint, FixedScalarKind, FullScalar, PastaCurve as PastaCurveAlt,
            ShortScalar,
        },
        FixedPoint as FixedPointConcrete, FixedPoints, ScalarFixed, NonIdentityPoint,
    },
    utilities::lookup_range_check::LookupRangeCheckConfig,
};

use halo2_proofs::{
    arithmetic::CurveAffine,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    pasta::{
        group::{
            cofactor::CofactorCurveAffine,
            ff::{PrimeField, WithSmallOrderMulGroup},
            Curve,
        },
        pallas, vesta,
    },
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Fixed, Instance, TableColumn},
    poly::{
        commitment::{Blind, Params, self},
        EvaluationDomain, LagrangeCoeff,
    },
};

use halo2_gadgets::ecc::chip::constants::find_zs_and_us;

trait PastaCurve: PastaCurveAlt {
    fn get_w_const_point() -> ConstPoint<Self>;
}

impl PastaCurve for pallas::Affine {
    fn get_w_const_point() -> ConstPoint<Self> {
        ConstPoint {
            gen: pallas_zs_and_us::generator(),
            us: pallas_zs_and_us::U.to_vec(),
            zs: pallas_zs_and_us::Z.to_vec(),
        }
    }
}

impl PastaCurve for vesta::Affine {
    fn get_w_const_point() -> ConstPoint<Self> {
        ConstPoint {
            gen: vesta_zs_and_us::generator(),
            us: vesta_zs_and_us::U.to_vec(),
            zs: vesta_zs_and_us::Z.to_vec(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct CurveTreeFixed<C: PastaCurve> {
    _ph: PhantomData<C>,
}

#[derive(Clone)]
struct ConstPoint<C: PastaCurve> {
    gen: C,
    us: Vec<[<C::Base as PrimeField>::Repr; 8]>, // todo store an array instead?
    zs: Vec<u64>, // todo same
}

impl<C: PastaCurve> Debug for ConstPoint<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.gen.fmt(f)
    }
}

impl<C: PastaCurve> PartialEq for ConstPoint<C> {
    fn eq(&self, other: &Self) -> bool {
        self.gen == other.gen
    }
}

impl<C: PastaCurve> Eq for ConstPoint<C> {
    fn assert_receiver_is_total_eq(&self) {}
}

impl<C: PastaCurve> ConstPoint<C> {
    fn new(gen: C) -> Self {
        let mut zs = vec![];
        let mut us = vec![];
        for (z, u) in find_zs_and_us(gen, 85).unwrap().into_iter() {
            zs.push(z);
            us.push(u.map(|e| e.to_repr()))
        }
        Self { gen, zs, us }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FullScalarConst<C: PastaCurve> {
    pnt: ConstPoint<C>,
}

struct EmptyFixedPoint<T: FixedScalarKind> {
    _ph: PhantomData<T>,
}

impl<T: FixedScalarKind> Debug for EmptyFixedPoint<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<T: FixedScalarKind> Clone for EmptyFixedPoint<T> {
    fn clone(&self) -> Self {
        Self { _ph: PhantomData }
    }
}

impl<T: FixedScalarKind> PartialEq for EmptyFixedPoint<T> {
    fn eq(&self, other: &Self) -> bool {
        true
    }
}

impl<T: FixedScalarKind> Eq for EmptyFixedPoint<T> {
    fn assert_receiver_is_total_eq(&self) {}
}

impl<C: PastaCurve> FixedPoint<C> for FullScalarConst<C> {
    type FixedScalarKind = FullScalar;

    fn generator(&self) -> C {
        self.pnt.gen
    }

    fn u(&self) -> Vec<[<C::Base as PrimeField>::Repr; 8]> {
        self.pnt.us.clone()
    }

    fn z(&self) -> Vec<u64> {
        self.pnt.zs.clone()
    }
}

impl<C: PastaCurve, T: FixedScalarKind> FixedPoint<C> for EmptyFixedPoint<T> {
    type FixedScalarKind = T;

    fn generator(&self) -> C {
        unreachable!()
    }

    fn u(&self) -> Vec<[<C::Base as PrimeField>::Repr; 8]> {
        unreachable!()
    }

    fn z(&self) -> Vec<u64> {
        unreachable!()
    }
}

impl<C: PastaCurve> FixedPoints<C> for CurveTreeFixed<C> {
    type Base = EmptyFixedPoint<BaseFieldElem>;
    type FullScalar = FullScalarConst<C>;
    type ShortScalar = EmptyFixedPoint<ShortScalar>;
}

// We want to represent a (possibly rerandomized) node in the tree which is blinded by a scalar `r', in a way that allows us to lookup one of the children.
// The idea is that the first of the two advice columns will have each (usable) index populated with the children of the node (i.e. their x-coordinates) and lookup enabled.
// However some number of the last rows (generators) are reserved to achieve zero knowledge. These will be blinded by random values chosen at proving time.
// To convince the verifier we introduce a second advice column which is the difference (for each scalar, and when seen as points) between the original commitment and the first advice column.
// This means that it should be zero in all the coordinates used to commit to the children, which we will enforce by a custom gate.
type CommittedAdvice = [Column<Advice>; 2];

#[derive(Debug, Clone)]
struct CurveTreeConfig<C: PastaCurve> {
    table: TableColumn,
    commitment: CommittedAdvice, // todo we probably need at least 2 in the final circuit.
    advices: [Column<Advice>; 10],
    instances: [Column<Instance>; 2],
    constants: [Column<Fixed>; 8],
    ecc_conf: EccConfig<C, CurveTreeFixed<C>>,
    range_check: LookupRangeCheckConfig<C::Base, 10>, // todo only needed for parts of ecc chip we are not using. Would we get better performance by cutting parts of the ecc chip?
}

#[derive(Debug, Default)]
struct CurveTreeCircuit<C: PastaCurve> {
    blind: Value<C::Scalar>,
    child: Value<C>,
}

impl<C: PastaCurve> Circuit<C::Base> for CurveTreeCircuit<C> {
    type Config = CurveTreeConfig<C>;
    type FloorPlanner = SimpleFloorPlanner;

    fn configure(meta: &mut ConstraintSystem<C::Base>) -> Self::Config {
        let commitment = [(); 2].map(|_| meta.advice_column());
        let advices = [(); 10].map(|_| meta.advice_column());
        let instances = [(); 2].map(|_| meta.instance_column());
        let constants = [(); 8].map(|_| meta.fixed_column());
        let table = meta.lookup_table_column();

        // todo do we enable equality on the commitment column?

        let range_check = LookupRangeCheckConfig::configure(meta, advices[0], table);
        let ecc_conf = EccChip::configure(meta, advices, constants, range_check);
        CurveTreeConfig {
            table,
            commitment,
            advices,
            instances,
            constants,
            ecc_conf,
            range_check,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<C::Base>,
    ) -> Result<(), Error> {
        let ecc_chip: EccChip<C, CurveTreeFixed<C>> = EccChip::construct(config.ecc_conf);

        let blind = ScalarFixed::new(ecc_chip.clone(), layouter.namespace(|| "blinding scalar"), self.blind)?; // todo why do we call namespace on layouter?
        
        // todo maybe ConstPoint should impl FullScalar instead.
        let blinding_base = FixedPointConcrete::from_inner(ecc_chip.clone(), FullScalarConst { pnt: C::get_w_const_point() });
        let (blinding_element, _) = blinding_base.mul(layouter.namespace(|| "blinding element"), blind)?;

        
        // todo add an index to the config
        // todo look up the x coordinate at the index (in a "committed advice colum")
        // maybe for now we just hardcode index 0 and make all other entries 0 

        // todo show that (x,y) is permissible (maybe make a separate chip) (also need sqrt in config, or compute while proving)
        let child_point = NonIdentityPoint::new(ecc_chip.clone(), layouter.namespace(|| "child point"), self.child)?;

        
        // add the blinding element to the point in the config.
        let rerandomized_child = blinding_element.add(layouter.namespace(|| "rerandomize child"), &child_point)?;

        // todo expose the rerandomized value (or constrain it to an input)



        Ok(())
    }

    fn without_witnesses(&self) -> Self {
        Self::default()
    }
}

pub fn main() {
    // todo figure out what k is needed
    let k = 10; // there will be n=2^k generators/rows in the proof.
    let n = 1 << k;
    // todo I believe we can just use n as branching factor.
    // it seems we need to leave some generators unused because of the permutation argument
    let pallas_params = halo2_proofs::poly::commitment::Params::<pallas::Affine>::new(k);
    assert_eq!(pallas_params.get_w(), pallas_zs_and_us::generator());
    let pallas_const_point = ConstPoint {
        gen: pallas_zs_and_us::generator(),
        us: pallas_zs_and_us::U.to_vec(),
        zs: pallas_zs_and_us::Z.to_vec(),
    };
    assert_eq!(pallas_const_point, pallas::Affine::get_w_const_point());
    
    
    let vesta_params = halo2_proofs::poly::commitment::Params::<vesta::Affine>::new(k);
    assert_eq!(vesta_params.get_w(), vesta_zs_and_us::generator());
    let vesta_const_point = ConstPoint {
        gen: vesta_zs_and_us::generator(),
        us: vesta_zs_and_us::U.to_vec(),
        zs: vesta_zs_and_us::Z.to_vec(),
    };
    assert_eq!(vesta_const_point, vesta::Affine::get_w_const_point());
}
