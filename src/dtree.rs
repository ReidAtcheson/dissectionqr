use crate::sparse::{NodeSet,Graph};




/// A tree node to hold the result of nested dissection
pub struct DissectionNode<S : NodeSet>{
    ///A `NodeSet` representing the columns for this node
    pub col : S,
    ///Level of the tree
    pub level : usize, 
    ///A unique index for this node
    pub id : usize,
    ///A vector of tree nodes representing rows for this node
    pub rows : Vec<usize>,
    ///A vector of tree nodes representing lower triangular (including diagonal) blocks
    pub lrows : Vec<usize>,
    ///A vector of tree nodes representing upper triangular (not including diagonal) blocks
    pub urows : Vec<usize>,
    ///The children of this node, if any
    pub children : Option< (usize,usize) >,
    ///The parent of this node, if any
    pub parent : Option<usize>

}

/// The tree of DissectionNodes
pub struct DissectionTree<S : NodeSet>{
    ///Index to root node
    pub root : usize,
    ///Vector of tree nodes
    pub arena : Vec<DissectionNode<S>>,
    ///Levels
    pub levels : Vec< Vec<usize> >
}


/// Builds basic nested dissection tree without level information
fn nested_dissection_basic<S : NodeSet + Clone ,G : Graph<S>>(s : &S, g : &G,maxnodes : usize,pathlen : usize) -> DissectionTree<S> {

    let root = DissectionNode { col : s.clone(), level : 0, id : 0, rows : vec![], lrows : vec![], urows : vec![], children : None, parent : None };
    let mut dtree = DissectionTree {root : 0, arena : vec![root], levels : vec![] };
    let mut stack : Vec<usize> = vec![0];

    //This builds the basic tree
    while stack.len()>0 {
        let j = stack.pop().unwrap();
        //Indices of children nodes
        let cid1 = dtree.arena.len();
        let cid2 = dtree.arena.len()+1;
        let level = dtree.arena[j].level;
        let id = dtree.arena[j].id;
        let p = &dtree.arena[j].col;
        let nnodes=p.nnodes();
        if nnodes>maxnodes{
            let (sep,p1,p2) = g.split_len(&p,pathlen).unwrap();
            //Current tree node becomes a separator
            dtree.arena[j].col=sep;

            //Current tree node gets children nodes
            dtree.arena[j].children = Some( (cid1,cid2) );
            //Create children nodes and place into arena
            let child1 = DissectionNode{col : p1, level : level+1, id : cid1, rows : vec![], lrows : vec![], urows : vec![], children : None, parent : Some(id)};
            let child2 = DissectionNode{col : p2, level : level+1, id : cid2, rows : vec![], lrows : vec![], urows : vec![],  children : None, parent : Some(id)};
            dtree.arena.push(child1);
            dtree.arena.push(child2);


            //Place children nodes onto stack
            stack.push(cid1);
            stack.push(cid2);
        }
    }
    dtree
}
/// After basic nested dissection tree is built we can construct the levels
/// of the tree.
fn build_levels<S : NodeSet>(dtree : &mut DissectionTree<S>) -> () {
    let nlevels=dtree.arena.iter().map(|x|{x.level}).max().unwrap()+1;
    dtree.levels=vec![vec![];nlevels];
    for x in dtree.arena.iter() {
        let level=x.level;
        let id=x.id;
        dtree.levels[level].push(id);
    }
}
// After building levels of the tree we can propagate upper and lower
// triangular blocks
fn lower_triangular_blocks<S : NodeSet>(dtree : &mut DissectionTree<S>) -> () {
    //Iterate from top level of tree to bottom level
    for level in dtree.levels.iter() {
        //Propagate to lower levels
        for id in level.iter() {
            dtree.arena[*id].rows.push(*id);
            match dtree.arena[*id].children {
                Some((c1,c2)) => {
                    for r in dtree.arena[*id].rows.clone().iter(){
                        dtree.arena[c1].rows.push(*r);
                        dtree.arena[c2].rows.push(*r);
                    }
                },
                None => ()
            }

        }
    }
}
fn upper_triangular_blocks<S : NodeSet>(dtree : &mut DissectionTree<S>) -> () {
    //Iterate from top level of tree to bottom level
    for level in dtree.levels.iter().rev() {
        //Propagate to lower levels
        for id in level.iter() {
            match dtree.arena[*id].parent {
                Some(p) => {
                    for r in dtree.arena[*id].rows.clone().iter(){
                        dtree.arena[p].rows.push(*r);
                    }
                },
                None => ()
            }
        }
    }
    //Sort all the row arrays for each node
    //and then remove duplicates
    for t in dtree.arena.iter_mut(){
        t.rows.sort_unstable();
        t.rows.dedup();
    }
}

/// Separate upper and lower triangular blocks
fn separate_upper_lower<S : NodeSet>(dtree : &mut DissectionTree<S>) -> () {
    for t in dtree.arena.iter_mut(){
        for r in t.rows.iter_mut(){
            let tid=t.id;
            //Lower triangular matrix
            let mut lrows = &mut t.lrows;
            //Upper triangular matrix
            let mut urows = &mut t.urows;
            if tid>=*r{
                lrows.push(*r);
            }
            else{
                urows.push(*r);
            }
        }
    }

    //Make sure `lrows` and `urows` are sorted
    for t in dtree.arena.iter_mut(){
        t.lrows.sort_unstable();
        t.urows.sort_unstable();
    }
}



pub fn nested_dissection<S : NodeSet + Clone , G : Graph<S> + Clone>(s : &S,g : &G,maxnodes : usize,pathlen : usize) -> DissectionTree<S> {
    let mut dtree = nested_dissection_basic(s,g,maxnodes,pathlen);
    build_levels(&mut dtree);
    lower_triangular_blocks(&mut dtree);
    upper_triangular_blocks(&mut dtree);
    separate_upper_lower(&mut dtree);
    dtree
}



pub fn graphviz<S : NodeSet>(dtree : &DissectionTree<S>) -> (){
    let mut stack : Vec<usize> = vec![dtree.root];

    print!("\n");
    print!("graph g{{\n");
    while stack.len()>0{
        let j=stack.pop().unwrap();
        let t=&dtree.arena[j];
        match t.children {
            Some((c1,c2)) => {
                print!("{} -> {};\n",t.id,c1);
                print!("{} -> {};\n",t.id,c2);
                stack.push(c1);
                stack.push(c2);
            },
            None => ()
        };
    }
    print!("}}\n");
    print!("\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    //use crate::sparse::*;
    use crate::grid::*;
    use std::collections::BTreeSet;

    #[test]
    fn nested_dissection_grid1d_levels_correct(){
        //Nodes within a level should correctly record their level
        let s = Grid1D{beg : 0, end : 1000};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        let dtree = {   
            let mut dtree = nested_dissection_basic(&s,&g,200,1);
            build_levels(&mut dtree);
            lower_triangular_blocks(&mut dtree);
            upper_triangular_blocks(&mut dtree);
            dtree
        };
        let arena=dtree.arena;
        for (id,level) in dtree.levels.iter().enumerate(){
            for n in level.iter(){
                assert_eq!(arena[*n].level,id);
            }
        }
    }

    #[test]
    fn nested_dissection_grid1d_parent_child_correct(){
        //Check that parent/child relationship is correct
        let s = Grid1D{beg : 0, end : 1000};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        let dtree = {   
            let mut dtree = nested_dissection_basic(&s,&g,200,1);
            build_levels(&mut dtree);
            lower_triangular_blocks(&mut dtree);
            upper_triangular_blocks(&mut dtree);
            dtree
        };
        let arena=dtree.arena;
        for (id,n) in arena.iter().enumerate(){
            let true_parent=n.id;
            assert_eq!(n.id,id);
            //Get children from node
            match n.children{
                Some ((c1,c2)) =>{
                    let p1=arena[c1].parent.unwrap();
                    let p2=arena[c2].parent.unwrap();
                    assert_eq!(p1,p2);
                    assert_eq!(true_parent,p1);
                    assert_eq!(true_parent,p2);
                },
                None => ()
            }
        }
    }




    #[test]
    fn nested_dissection_grid1d_rows_sorted(){
        let s = Grid1D{beg : 0, end : 1000};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        let dtree = {   
            let mut dtree = nested_dissection_basic(&s,&g,200,1);
            build_levels(&mut dtree);
            lower_triangular_blocks(&mut dtree);
            upper_triangular_blocks(&mut dtree);
            dtree
        };
        //The rows of every tree-node should be sorted
        for n in dtree.arena.iter(){
            let rows = &n.rows;
            let mut sorted=true;
            for i in 0..rows.len()-1{
                sorted=sorted&&(rows[i]<rows[i+1]);
            }
            //Note that because of the strict
            //inequality above this also guarantees
            //there are no duplicates in the `rows` 
            //array.
            assert!(sorted);
        }
    }



    //Parent rows should be union of children rows
    #[test]
    fn nested_dissection_grid1d_parent_child_union(){
        let dtree = {   
            let s = Grid1D{beg : 0, end : 1000};
            let g = Stencil1D{ offsets : vec![-1,0,1] };
            let mut dtree = nested_dissection_basic(&s,&g,200,1);
            build_levels(&mut dtree);
            lower_triangular_blocks(&mut dtree);
            upper_triangular_blocks(&mut dtree);
            dtree
        };
        for n in dtree.arena.iter(){
            let parent = n.rows.clone();
            match n.children{
                Some((c1,c2)) => {
                    let child1=dtree.arena[c1].rows.clone();
                    let child2=dtree.arena[c2].rows.clone();
                    let p : BTreeSet<usize> = parent.iter().cloned().collect();
                    let s2 : BTreeSet<usize> = child1.iter().cloned().collect();
                    let s3 : BTreeSet<usize> = child2.iter().cloned().collect();
                    let s : BTreeSet<usize> = s2.union(&s3).cloned().collect();
                    assert!(p == s);
                },
                None => ()
            }
        }
    }

    #[test]
    fn nested_dissection_grid1d_level_order(){
        let dtree = {   
            let s = Grid1D{beg : 0, end : 1000};
            let g = Stencil1D{ offsets : vec![-1,0,1] };
            let mut dtree = nested_dissection_basic(&s,&g,200,1);
            build_levels(&mut dtree);
            lower_triangular_blocks(&mut dtree);
            upper_triangular_blocks(&mut dtree);
            dtree
        }; 
        let nlevels=dtree.arena.iter().map(|x|{x.level}).max().unwrap()+1;
        assert_eq!(dtree.levels[0].len(),1);
        assert_eq!(dtree.arena[dtree.levels[0][0]].level,0);
        for node in dtree.levels[dtree.levels.len()-1].iter(){
            assert_eq!(dtree.arena[*node].level,nlevels-1);
        }
    }








    /*
    #[test]
    fn nested_dissection_graphviz(){
        let s = Grid1D{beg : 0, end : 1000};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        let dtree = nested_dissection_basic(&s,&g,200,1);
        graphviz(&dtree);
    }
    */
}



