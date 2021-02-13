
use crate::sparse::*;
use crate::dtree::*;
use crate::utility::*;
use std::collections::BTreeMap;
use ndarray::prelude::Ix2;
use ndarray::{Array,Array2};
use ndarray::ShapeBuilder;
use crate::utility::*;
use num_traits::Float;





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

fn gather<F : Float>(rids : &Vec<usize>,rows : &BTreeMap<usize,Array2<F>>) -> Array2<F>{

    let ncols=rows.get(&rids[0]).unwrap().shape()[1];
    let nrows=rids.iter().map(|x| rows.get(x).unwrap().shape()[0] ).fold(0,|acc,x| acc+x);
    let m=50;
    let a : Array2<F> = Array::zeros((m,m).f());
    a
}

/// Performs numeric factorization
/// This proceeds in the simplest fashion:
/// It factorizes one level at a time starting at the bottom most level
/// proceeding like this until top node has been factorized
/// Once top node has been factorized, the algorithm is complete.
fn factorize<F : Float, S : NodeSet>(fact : &mut NumericFactorization<F,S>) -> () {
    let levels = &fact.dtree.levels;
    let arena  = &fact.dtree.arena;
    let nblocks=arena.len();

    //We want to start at the bottom level, that is 
    //why we reverse levels
    for level in levels.iter().rev(){
        for cid in level.iter(){
            //Stack together all the assembled submatrices
            //in lower triangular part of overall matrix
            //then QR factorize the result
            let rids = {
                let mut rids = Vec::<usize>::new();
                for r in arena[*cid].rows.iter(){
                    if r>=cid{
                        rids.push(*r);
                    }
                }
                rids
            };
            let rows = &fact.data[*cid];
            let stacked = gather(&rids,&rows);
            let qr = QRFact::<F>::new(stacked);
        }
    }
}
