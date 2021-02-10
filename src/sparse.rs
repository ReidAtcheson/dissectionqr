
use ndarray::prelude::Ix2;
use ndarray::Array;

///A generic set of nodes with the minimal functions necessary for sparse matrix manipulation
pub trait NodeSet {
    ///Determines if an input node, represented as an integer is contained in this nodeset
    fn contains(&self,i : i64) -> bool;
    ///Determines if input nodeset is disjoint from this nodeset
    fn disjoint(&self,s : Self) -> bool;
    ///Represents this nodeset as vector of integers
    fn enumerate(&self) -> Vec<i64>;
    ///Gives number of nodes contained in this nodeset
    fn nnodes(&self) -> usize;
}

///Defines the sparsity pattern for a matrix
///Edges in this graph (which may be implicit and not necessarily stored in memory) 
///represent nonzero entries in a sparse matrix
pub trait Graph<S : NodeSet>{
    ///Gets the largest nodeset this graph is defined on
    fn superset(&self) -> S;
    ///Determines if `s1` is reachable from `s2` using edges from this graph
    fn reachable(&self,s1 : S,s2 : S) -> bool;
    ///Determines if `s1` is reachable from `s2` by a path of length `len` using edges from this
    ///graph
    fn reachable_len(&self,s1 : S,s2 : S,len : usize) -> bool;
    ///Split an input nodeset into three nodesets: two partitions and a separator such that the two partitions 
    ///are not reachable from each other.
    fn split(&self, s : S) -> (S,S,S);
    ///Split an input nodeset into three nodesets: two partitions and a separator such that the two
    ///partitions are not reachable by a specified path length `len`
    fn split_len(&self,s : S,len : usize) -> (S,S,S);
}

///Defines basic sparse matrix. Indexed on nodes from a `NodeSet` and nonzero entries
///defined by edges from a `Graph`. The type `F` is the underlying field.
pub trait SparseMatrix<F, S : NodeSet, G : Graph<S>>{
    ///Given row and column indices assemble the resulting submatrix
    ///from this sparse matrix into a _dense_ matrix. 
    ///This is used for assembling supernodes on a nested dissection tree,
    ///not for true sparse matrix assembly.
    fn assemble(&self,rows : S,cols : S) -> Array<F,Ix2>;
}


