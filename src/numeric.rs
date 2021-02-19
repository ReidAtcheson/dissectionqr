use crate::sparse::*;
use crate::dtree::*;
use crate::utility::*;
use std::collections::BTreeMap;
use ndarray::prelude::*;
//use ndarray::{Array,Array2};
//use ndarray::ShapeBuilder;
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
pub fn assemble<F : Clone, S : NodeSet, G : Graph<S>, M : SparseMatrix<F,S,G>>(dtree : DissectionTree<S>,op : M ) -> NumericFactorization<F,S> {

    let nblocks=dtree.levels.iter().map(|x| x.iter().map(|y| *y).max().unwrap()).max().unwrap()+1;
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
    let mut a : Array2<F> = Array::zeros((nrows,ncols).f());
    let mut offs : usize = 0;
    for r in rids.iter(){
        let omat = &rows.get(r).expect("No matrix for matching key in the temporary dictionary");
        match omat {
            Some(mat) =>{
                let shape=mat.shape();
                let nrows=shape[0];
                let ncols=shape[1];
                let mut slice_a=a.slice_mut(s![offs..offs+nrows,..]);
                //Copy mat into a
                for (u,v) in slice_a.iter_mut().zip(mat.iter()){
                    *u=v.clone();
                }

                offs+=nrows;
            },
            _ => panic!("No matrix allocated in the temporary dictionary despite matching key")
        }
    }
    a
}
fn scatter<F : Lapack<F> + Clone + Zero >(rids : &Vec<usize>,rows : &mut BTreeMap<usize,Option<Array2<F>>>,stacked : &Array2<F> ) ->(){
    let a=stacked;
    let ncols=rows.get(&rids[0]).unwrap().as_ref().unwrap().shape()[1];
    let nrows=rids.iter().map(|x| rows.get(x).unwrap().as_ref().unwrap().shape()[0] ).fold(0,|acc,x| acc+x);
    if ncols != a.shape()[1]{
        panic!("Number of columns in BTreeMap matrices does not match input stacked matrix");
    }
    if nrows != a.shape()[0]{
        panic!("Number of rows in BTreeMap matrices does not match input stacked matrix");
    }

    for r in rids.iter(){
        let mut offs : usize =0;
        match *rows.get_mut(r).unwrap() {
            Some(ref mut mat) =>{
                let shape=mat.shape();
                let nrows=shape[0];
                let ncols=shape[1];
                let slice_a=a.slice(s![offs..offs+nrows,..]);
                //Copy a into mat
                for (u,v) in slice_a.iter().zip(mat.iter_mut()){
                    *v=u.clone();
                }

                offs+=nrows;
            },
            _ => panic!("No matrix allocated in the temporary dictionary despite matching key")
        }
    }
}







/// Performs numeric factorization
/// This proceeds in the simplest fashion:
/// It factorizes one level at a time starting at the bottom-most level
/// proceeding like this until top node has been factorized
/// Once top node has been factorized, the algorithm is complete.
pub fn factorize<F : Lapack<F> + Clone + Zero, S : NodeSet>(fact : &mut NumericFactorization<F,S>) -> () {
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
            let stacked = {
                let data = &fact.data[*cid];
                let stacked = gather(&n.lrows,&data);
                stacked
            };
            let mut qr = QRFact::<F>::new(stacked);
            //Now iterate up the parents and apply Q^T until top node 
            let mut parent = n.parent;
            while let Some(p) = parent{
                let pn = &fact.dtree.arena[p];
                let pdata = &fact.data[p];
                let mut pstacked = gather(&n.lrows,&pdata);
                pstacked=qr.mul_left_qt(pstacked);
                scatter(&n.lrows,&mut fact.data[p],&pstacked);
                parent=pn.parent;
            }
            fact.qrs[*cid]=Some(qr);
            //With these columns processed we no longer need the flexible storage
            //of the BTreeMap so I gather the upper triangular part into a single
            //stacked matrix and then set the entire BTreeMap to None
            let ru = {
                let data = &fact.data[*cid];
                gather(&n.urows,&data)
            };
            fact.rus[*cid]=Some(ru);
            //Set the BTreeMap to None, we no longer need it for these columns.
            for (key,value) in fact.data[*cid].iter_mut(){
                *value=None;
            }
        }
    }
}



impl <F : Lapack<F> + Clone, S : NodeSet>  NumericFactorization<F,S>{
    pub fn solve(&mut self,b : Array2<F>) -> Array2<F>{
        b
    }
}
