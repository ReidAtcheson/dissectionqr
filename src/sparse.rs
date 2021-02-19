
use ndarray::prelude::Ix2;
use ndarray::{Array,Array2};


///A generic set of nodes with the minimal functions necessary for sparse matrix manipulation
pub trait NodeSet {
    ///Determines if an input node, represented as an integer is contained in this nodeset
    fn contains(&self,i : i64) -> bool;
    ///Determines if input nodeset is disjoint from this nodeset
    fn disjoint(&self,s : &Self) -> bool;
    ///Determines if this nodeset is a subset of input subset
    fn subset(&self,s : &Self) -> bool;
    ///Represents this nodeset as vector of integers
    fn enumerate(&self) -> Vec<i64>;
    ///Gives number of nodes contained in this nodeset
    fn nnodes(&self) -> usize;
}

///Defines the sparsity pattern for a matrix
///Edges in this graph (which may be implicit and not necessarily stored in memory) 
///represent nonzero entries in a sparse matrix
pub trait Graph<S : NodeSet>
{

    fn reachable_len_oneway(&self, s1 : &S, s2 : &S, len : usize) -> bool;
    ///Determines if `s1` is reachable from `s2` by a path of length `len` using edges from this
    ///graph and vice-versa
    fn reachable_len(&self,s1 : &S,s2 : &S,len : usize) -> bool{
        self.reachable_len_oneway(s1,s2,len) && self.reachable_len_oneway(s2,s1,len)
    }
    ///Determines if `s1` is reachable from `s2` using edges from this graph
    fn reachable(&self,s1 : &S,s2 : &S) -> bool{
        self.reachable_len(s1,s2,1)
    }
    ///Split an input nodeset into three nodesets: two partitions and a separator such that the two
    ///partitions are not reachable by a specified path length `len`
    fn split_len(&self,s : &S,len : usize) -> Option<(S,S,S)>;

    ///Split an input nodeset into three nodesets: two partitions and a separator such that the two partitions 
    ///are not reachable from each other.
    fn split(&self, s : &S) -> Option<(S,S,S)>{
        self.split_len(s,1)
    }
}

///Defines basic sparse matrix. Indexed on nodes from a `NodeSet` and nonzero entries
///defined by edges from a `Graph`. The type `F` is the underlying field.
pub trait SparseMatrix<F, S : NodeSet, G : Graph<S>>{

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    ///Given row and column indices assemble the resulting submatrix
    ///from this sparse matrix into a _dense_ matrix. 
    ///This is used for assembling supernodes on a nested dissection tree,
    ///not for true sparse matrix assembly.
    fn assemble(&self,rows : &S,cols : &S) -> Array2<F>;

    fn eval(&self,x : &Array2<F>,y : &mut Array2<F>) -> ();

}


