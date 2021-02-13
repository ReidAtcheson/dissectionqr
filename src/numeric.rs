
use crate::sparse::*;
use crate::dtree::*;
use crate::utility::*;
use std::collections::BTreeMap;
use ndarray::prelude::Ix2;
use ndarray::Array2;
use crate::utility::*;





pub struct NumericFactorization<F, S : NodeSet> {
    dtree : DissectionTree<S>,
    data : Vec<BTreeMap<usize,Array2<F>>>,
    qrs  : Vec<QRFact<F>>,
    rus  : Vec<Array2<F>>
}

/// Assembles the matrix represented by `op` into numeric factorization datastructure
/// in preparation for factorization
fn assemble<F : Clone, S : NodeSet, G : Graph<S>, M : SparseMatrix<F,S,G>>(dtree : DissectionTree<S>,op : M ) -> NumericFactorization<F,S> {

    let nblocks=dtree.levels.iter().map(|x| x.iter().map(|y| *y).max().unwrap()).max().unwrap();
    let mut data : Vec<BTreeMap<usize,Array2<F>>> = vec![ BTreeMap::<usize,Array2<F>>::new() ; nblocks ];
    for level in dtree.levels.iter(){
        for cid in level.iter(){
            for rid in dtree.arena[*cid].rows.iter(){
                let row=&dtree.arena[*rid].col;
                let col=&dtree.arena[*cid].col;
                data[*cid].insert(*rid,op.assemble(row,col));
            }
        }
    }
    NumericFactorization { dtree : dtree,data : data, qrs : vec![], rus : vec![] }
}

/// Performs numeric factorization
fn factorize<F : Clone, S : NodeSet>(fact : &mut NumericFactorization<F,S>) -> () {
    ()
}
