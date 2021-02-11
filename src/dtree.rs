use crate::sparse::*;




/// A tree to hold the result of nested dissection
pub struct DissectionNode<S : NodeSet>{
    ///A `NodeSet` representing the columns for this node
    col : S,
    ///Level of the tree
    level : usize, 
    ///A unique index for this node
    id : usize,
    ///A vector of tree nodes representing rows for this node
    rows : Vec<usize>,
    ///The children of this node, if any
    children : Option< (usize,usize) >,
    ///The parent of this node, if any
    parent : Option<usize>

}

pub struct DissectionTree<S : NodeSet>{
    ///Index to root node
    root : usize,
    ///Vector of tree nodes
    arena : Vec<DissectionNode<S>>,
    ///Levels
    levels : Vec< Vec<usize> >
}


/// Builds basic nested dissection tree without level information
fn nested_dissection<S : NodeSet + Clone ,G : Graph<S>>(s : &S, g : &G,maxnodes : usize,pathlen : usize) -> DissectionTree<S> {

    let mut id : usize = 0;
    let root = DissectionNode { col : s.clone(), level : 0, id : id, rows : vec![], children : None, parent : None };
    let mut dtree = DissectionTree {root : id, arena : vec![root], levels : vec![] };
    let mut stack : Vec<usize> = vec![id];

    //This builds the basic tree
    while stack.len()>0 {
        let j = stack.pop().unwrap();
        let t = &mut dtree.arena[j];
        let p = &t.col;
        let nnodes=p.nnodes();
        if nnodes>maxnodes{
            let (sep,p1,p2) = g.split_len(&p,pathlen).unwrap();
            //Current tree becomes a separator
            t.col=sep;
            //Current tree gets children nodes - start with their ids
            t.children = Some( (id+1,id+2) );
            //Place children nodes onto stack
            stack.push(id+1);
            stack.push(id+2);

            //Create children nodes and place into arena
            let child1 = DissectionNode{col : p1, level : t.level+1, id : id+1, rows : vec![], children : None, parent : Some(id)};
            let child2 = DissectionNode{col : p2, level : t.level+1, id : id+2, rows : vec![], children : None, parent : Some(id)};
            dtree.arena.push(child1);
            dtree.arena.push(child2);
            id+=2;
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
                        if (*r)>(*id){
                            dtree.arena[p].rows.push(*r);
                        }
                    }
                },
                None => ()
            }
        }
    }
    //Sort all the row arrays for each node
    for t in dtree.arena.iter_mut(){
        t.rows.sort_unstable();
    }
}



fn graphviz<S : NodeSet>(dtree : &DissectionTree<S>) -> (){
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
    use crate::sparse::*;
    use crate::grid::*;
    use std::collections::BTreeSet;

    #[test]
    fn nested_dissection_grid1d_levels_correct(){
        //Nodes within a level should correctly record their level
        let s = Grid1D{beg : 0, end : 1000};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        let dtree = {   
            let mut dtree = nested_dissection(&s,&g,200,1);
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
            let mut dtree = nested_dissection(&s,&g,200,1);
            build_levels(&mut dtree);
            lower_triangular_blocks(&mut dtree);
            upper_triangular_blocks(&mut dtree);
            dtree
        };
        let arena=dtree.arena;
        for n in arena.iter(){
            let true_parent=n.id;
            //Get children from node
            match n.children{
                Some ((c1,c2)) =>{
                    let p1=arena[c1].parent.unwrap();
                    let p2=arena[c2].parent.unwrap();
                    print!("\np1:  {},    p2:    {},   tp:   {}\n",p1,p2,true_parent);
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
            let mut dtree = nested_dissection(&s,&g,200,1);
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
        let s = Grid1D{beg : 0, end : 1000};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        let dtree = {   
            let mut dtree = nested_dissection(&s,&g,200,1);
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
                    let P : BTreeSet<usize> = parent.iter().cloned().collect();
                    let S2 : BTreeSet<usize> = child1.iter().cloned().collect();
                    let S3 : BTreeSet<usize> = child2.iter().cloned().collect();
                    let S : BTreeSet<usize> = S2.union(&S3).cloned().collect();
                    print!("parent {:?}\n",parent);
                    print!("child1 {:?}\n",child1);
                    print!("child2 {:?}\n",child2);
                    assert!(P == S);
                },
                None => ()
            }
        }
    }









    /* Below is for printing out the tree for inspection but not a real test
    #[test]
    fn nested_dissection_graphviz(){
        let s = Grid1D{beg : 0, end : 1000};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        let dtree = nested_dissection(&s,&g,200,1);
        graphviz(&dtree);
    }
    */
}



