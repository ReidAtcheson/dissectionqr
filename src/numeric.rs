
use crate::sparse::*;
use crate::dtree::*;
use std::collections::BTreeMap;
use ndarray::prelude::Ix2;
use ndarray::Array2;





pub struct NumericFactorization<F, S : NodeSet> {
    dtree : DissectionTree<S>,
    data : Vec<BTreeMap<usize,Array2<F>>>,
    qrs  : Array2<F>,
    rus  : Array2<F>
}
