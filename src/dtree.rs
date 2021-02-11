use crate::sparse::*;
use crate::grid::*;




/// A tree to hold the result of nested dissection
pub struct DissectionNode<S : NodeSet>{
    ///A `NodeSet` representing the columns for this node
    col : S,
    ///Level of the tree
    level : usize, 
    ///A unique index for this node
    id : usize,
    ///A vector of `NodeSet`s representing rows for this node
    rows : Vec<S>,
    ///The children of this node, if any
    children : Option< (usize,usize) >,
    ///The parent of this node, if any
    parent : Option<usize>

}

pub struct DissectionTree<S : NodeSet>{
    ///Index to root node
    root : usize,
    ///Vector of tree nodes
    arena : Vec<DissectionNode<S>> 
}


fn nested_dissection<S : NodeSet + Clone ,G : Graph<S>>(s : &S, g : &G,maxnodes : usize,pathlen : usize) -> DissectionTree<S> {

    let mut id : usize = 0;
    let root = DissectionNode { col : s.clone(), level : 0, id : id, rows : vec![], children : None, parent : None };
    let mut dtree = DissectionTree {root : id, arena : vec![root]};
    let mut stack : Vec<usize> = vec![id];

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



