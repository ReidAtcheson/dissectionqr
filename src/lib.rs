use ndarray::prelude::*;
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
///defined by edges from a `Graph`
pub trait SparseMatrix<F, S : NodeSet, G : Graph<S>>{
    ///Given row and column indices assemble the resulting submatrix
    ///from this sparse matrix into a _dense_ matrix. 
    ///This is used for assembling supernodes on a nested dissection tree,
    ///not for true sparse matrix assembly.
    fn assemble(&self,rows : S,cols : S) -> Array<F,Ix2>;
}


#[derive(Eq,PartialEq,Ord,PartialOrd,Clone,Copy,Hash,Debug)]
pub struct Grid1D{
    beg : i64,
    end : i64
}


impl NodeSet for Grid1D{
    fn contains(&self,i : i64) -> bool {
        self.beg<=i && i<self.end
    }
    fn disjoint(&self,s : Self) -> bool {
        self.beg>=s.end || s.beg>=self.end
    }
    fn enumerate(&self) -> Vec<i64> {
        let out = (self.beg..self.end).collect::<Vec<i64>>();
        out
    }
    fn nnodes(&self) -> usize {
        (self.end-self.beg) as usize
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    //Sanity check that checks that an enumerated nodeset contains all its nodes
    fn nodeset_enumerate_all_in<S : NodeSet>(s : S){
        let nodes = s.enumerate();
        let all_in = nodes.iter().map(|x|{s.contains(*x)}).fold(false,|acc,x|{acc||x});
        assert!(all_in)
    }
    //Sanity check that checks the size of an enumerated nodeset is the same as its number
    //of nodes
    fn nodeset_length_equals_nnodes<S : NodeSet>(s : S) {
        assert_eq!( s.enumerate().len(), s.nnodes() )
    }

    #[test]
    fn grid1d_sanity() {
        let g = Grid1D{ beg : 0,end : 10};
        nodeset_enumerate_all_in(g);
        nodeset_length_equals_nnodes(g);
    }
    #[test]
    fn grid1d_disjoint_works(){
        let g1 = Grid1D{beg : 5, end : 10};
        let g2 = Grid1D{beg : 11, end : 20};
        let g3 = Grid1D{beg : 1, end : 4};

        assert!(g1.disjoint(g2));
        assert!(g1.disjoint(g3));
        assert!(g2.disjoint(g3));
    }
}
