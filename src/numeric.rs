
use crate::sparse::*;
use crate::dtree::*;
use crate::utility::*;
use std::collections::BTreeMap;
use ndarray::prelude::Ix2;
use ndarray::{Array,Array2};
use ndarray::ShapeBuilder;
use crate::utility::*;
use num_traits::{Float,Zero};





pub struct NumericFactorization<F, S : NodeSet> {
    dtree : DissectionTree<S>,
    data : Vec<BTreeMap<usize,Option<Array2<F>>>>,
    qrs  : Vec<Option<QRFact<F>>>,
    rus  : Vec<Option<Array2<F>>>
}

/// Assembles the matrix represented by `op` into numeric factorization datastructure
/// in preparation for factorization
fn assemble<F : Clone, S : NodeSet, G : Graph<S>, M : SparseMatrix<F,S,G>>(dtree : DissectionTree<S>,op : M ) -> NumericFactorization<F,S> {

    let nblocks=dtree.levels.iter().map(|x| x.iter().map(|y| *y).max().unwrap()).max().unwrap();
    let mut data : Vec<BTreeMap<usize,Option<Array2<F>>>> = vec![ BTreeMap::<usize,Option<Array2<F>>>::new() ; nblocks ];
    for level in dtree.levels.iter(){
        for cid in level.iter(){
            for rid in dtree.arena[*cid].rows.iter(){
                let row=&dtree.arena[*rid].col;
                let col=&dtree.arena[*cid].col;
                data[*cid].insert(*rid,Some(op.assemble(row,col)));
            }
        }
    }
    let qrs : Vec<Option<QRFact<F>>> = (0..nblocks).map(|x| None).collect();
    let rus : Vec<Option<Array2<F>>> = (0..nblocks).map(|x| None).collect(); 
    NumericFactorization { dtree : dtree,data : data, qrs : qrs, rus : rus }
}

fn gather<F : Lapack<F> + Clone + Zero >(rids : &Vec<usize>,rows : &BTreeMap<usize,Option<Array2<F>>>) -> Array2<F>{

    let ncols=rows.get(&rids[0]).unwrap().as_ref().unwrap().shape()[1];
    let nrows=rids.iter().map(|x| rows.get(x).unwrap().as_ref().unwrap().shape()[0] ).fold(0,|acc,x| acc+x);
    let m=50;
    let a : Array2<F> = Array::zeros((m,m).f());
    panic!("Gather not implemented yet");
    a
}
fn scatter<F : Lapack<F> + Clone + Zero >(rids : &Vec<usize>,rows : &mut BTreeMap<usize,Option<Array2<F>>>,stacked : &Array2<F> ) ->(){
    panic!("Scatter not implemented yet");
}







/// Performs numeric factorization
/// This proceeds in the simplest fashion:
/// It factorizes one level at a time starting at the bottom most level
/// proceeding like this until top node has been factorized
/// Once top node has been factorized, the algorithm is complete.
fn factorize<F : Lapack<F> + Clone + Zero, S : NodeSet>(fact : &mut NumericFactorization<F,S>) -> () {
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
            let n = &fact.dtree.arena[*cid];
            let data = &fact.data[*cid];
            let stacked = gather(&n.lrows,&data);
            let mut qr = QRFact::<F>::new(stacked);
            //Now iterate up the parents and apply Q^T until top node 
            let mut parent = n.parent;
            while let Some(p) = parent{
                let pn = &fact.dtree.arena[p];
                let mut pdata = &fact.data[p];
                let mut pstacked = gather(&n.lrows,&pdata);
                pstacked=qr.mul_left_qt(pstacked);
                scatter(&n.lrows,&mut pdata,&pstacked);

                parent=pn.parent;
            }
            fact.qrs[*cid]=Some(qr);
        }
    }
}
