
use crate::sparse::*;
///Set of nodes defined implicitly as all integers between
///`beg` and `end` not including `end`
#[derive(Eq,PartialEq,Ord,PartialOrd,Clone,Copy,Hash,Debug)]
pub struct Grid1D{
    pub beg : i64,
    pub end : i64
}


///Graph defined by a 1D stencil. The stencil offsets
///are contained in a vector.
#[derive(Eq,PartialEq,Ord,PartialOrd,Clone,Hash,Debug)]
pub struct Stencil1D{
    pub offsets : Vec<i64>
}

impl Grid1D{
    fn isvalid(&self) -> bool {
        self.beg<self.end
    }
}

impl NodeSet for Grid1D{
    fn contains(&self,i : i64) -> bool {
        self.beg<=i && i<self.end
    }
    fn disjoint(&self,s : &Self) -> bool {
        self.beg>=s.end || s.beg>=self.end
    }
    fn subset(&self,s : &Self) -> bool {
        self.beg>=s.beg && self.end<=s.end
    }
    fn enumerate(&self) -> Vec<i64> {
        let out = (self.beg..self.end).collect::<Vec<i64>>();
        out
    }
    fn nnodes(&self) -> usize {
        (self.end-self.beg) as usize
    }
}

impl Graph<Grid1D> for Stencil1D{
    fn reachable_len_oneway(&self,s1 : &Grid1D,s2 : &Grid1D,len : usize) -> bool{
        let slen = len as i64;
        let offsets = &self.offsets;
        //Expand input grids by stencil offsets
        let max_offs=offsets.iter().max().unwrap(); 
        let min_offs=offsets.iter().min().unwrap();
        let g1 = Grid1D{ beg : s1.beg + slen*min_offs, end : s1.end + slen*max_offs};
        //If intersection is nonempty, sets are reachable
        !g1.disjoint(&s2)
    }
    fn split_len(&self,s : &Grid1D,len : usize) -> Option<(Grid1D,Grid1D,Grid1D)>{
        let slen = len as i64;
        let max_offs=self.offsets.iter().max().unwrap();
        let min_offs=self.offsets.iter().min().unwrap();
        let m=std::cmp::max(max_offs.abs(),min_offs.abs());
        let nnodes = s.nnodes() as i64;
        let g1 =  Grid1D{ beg : s.beg                  , end : s.beg + nnodes/2 -slen*m};
        let g2 =  Grid1D{ beg : s.beg+nnodes/2         , end : s.end                   };
        let sep = Grid1D{ beg : s.beg + nnodes/2 - slen*m      , end : s.beg + nnodes/2                };

        if g1.isvalid() && g2.isvalid() && sep.isvalid(){
            Some((sep,g1,g2))
        }
        else{
            None
        }

    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
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

    fn graph_split_reachable<S : NodeSet, G : Graph<S>>(s : S,g : G){
        let len=2;
        let (sep,p1,p2) = g.split_len(&s,len).unwrap();


        assert!( !g.reachable_len(&p1,&p2,len) );
        assert!( !g.reachable_len(&p2,&p1,len) );
        assert!( sep.disjoint(&p1) );
        assert!( sep.disjoint(&p2) );
        assert!( p1.disjoint(&p2) );
    }


    #[test]
    fn stencil1d_reachable(){
        let s1 = Grid1D{ beg : 0,end : 10};
        let s2 = Grid1D{ beg : 10,end : 15};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        assert!( g.reachable(&s1,&s2) );
    }

    #[test]
    fn stencil1d_split_toosmall(){
        let s = Grid1D{ beg : 0,end : 10};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        assert!( g.split_len(&s,5) == None );
    }

    #[test]
    fn grid1d_sanity() {
        let g = Grid1D{ beg : 0,end : 10};
        nodeset_enumerate_all_in(g);
        nodeset_length_equals_nnodes(g);
    }

    #[test]
    fn grid1d_split_reachable() {
        let s = Grid1D{ beg : 0,end : 100};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        graph_split_reachable(s,g);
    }

    #[test]
    fn stencil1d_split_partition() {
        let s = Grid1D{ beg : 0,end : 100};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        let (sep,p1,p2) = g.split_len(&s,3).unwrap();

        assert!(p1.subset(&s));
        assert!(p2.subset(&s));
        assert!(sep.subset(&s));

        assert_eq!(p1.nnodes()+p2.nnodes()+sep.nnodes(),s.nnodes());

    }

    #[test]
    fn stencil1d_split_nonzero_start(){
        let s = Grid1D{ beg : 50,end : 100};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        let res = g.split_len(&s,3);
        assert!( res != None );
    }

    #[test]
    fn grid1d_subset(){
        let s1 = Grid1D{ beg : 50,end : 100};
        let s2 = Grid1D{ beg : 50,end : 100};
        let s3 = Grid1D{ beg : 70,end : 100};
        let s4 = Grid1D{ beg : 100,end : 110};
        let s5 = Grid1D{ beg : 0, end : 30};
        let s6 = Grid1D{ beg : 30, end : 100 };
        assert!( s1.subset(&s2) );
        assert!( !s1.subset(&s3) );
        assert!( !s1.subset(&s4) );
        assert!( !s1.subset(&s5) );
        assert!( s1.subset(&s6) );
    }








    #[test]
    fn grid1d_disjoint_works(){
        let g1 = Grid1D{beg : 5, end : 10};
        let g2 = Grid1D{beg : 11, end : 20};
        let g3 = Grid1D{beg : 1, end : 4};
        let g4 = Grid1D{beg : 0, end : 50};

        assert!(g1.disjoint(&g2));
        assert!(g1.disjoint(&g3));
        assert!(g2.disjoint(&g3));

        assert!(!g1.disjoint(&g4));
        assert!(!g2.disjoint(&g4));
        assert!(!g3.disjoint(&g4));

    }
}
