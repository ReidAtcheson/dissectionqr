use crate::sparse::*;


/// A tree to hold the result of nested dissection
pub struct DissectionTree<S : NodeSet>{
    ///A `NodeSet` representing the columns for this node
    col : S,
    ///Level of the tree
    level : i64, 
    ///A unique index for this node
    id : i64,
    ///A vector of `NodeSet`s representing rows for this node
    rows : Vec<S>,
    ///The children of this node, if any
    children : Option< (Box<DissectionTree<S>>,Box<DissectionTree<S>>) >
}


fn nested_dissection<S : NodeSet + Clone ,G : Graph<S>>(s : S) -> DissectionTree<S> {
    let q = s.clone();
    DissectionTree {col : s, level : 0, id : 0, rows : vec![q], children : None}
}
