
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

impl Graph<Grid1D> for Stencil1D{
    fn reachable(&self,s1 : Grid1D ,s2 : Grid1D) -> bool{
        let offsets = &self.offsets;
        //Expand input grids by stencil offsets
        let max_offs=match offsets.iter().max() {
            Some(x) => x,
            None => panic!()
        };
        let min_offs=match offsets.iter().min() {
            Some(x) => x,
            None => panic!()
        };
        let g1 = Grid1D{ beg : s1.beg + min_offs, end : s1.end + max_offs};
        let g2 = Grid1D{ beg : s2.beg + min_offs, end : s2.end + max_offs};
        //If intersection is nonempty, sets are reachable
        !g1.disjoint(g2)
    }
    fn reachable_len(&self,s1 : Grid1D,s2 : Grid1D,len : usize) -> bool{
        let slen = len as i64;
        let offsets = &self.offsets;
        //Expand input grids by stencil offsets
        let max_offs=match offsets.iter().max() {
            Some(x) => x,
            None => panic!()
        };
        let min_offs=match offsets.iter().min() {
            Some(x) => x,
            None => panic!()
        };
        let g1 = Grid1D{ beg : s1.beg + slen*min_offs, end : s1.end + slen*max_offs};
        let g2 = Grid1D{ beg : s2.beg + slen*min_offs, end : s2.end + slen*max_offs};
        //If intersection is nonempty, sets are reachable
        !g1.disjoint(g2)
    }
    fn split_len(&self,s : Grid1D,len : usize) -> Option<(Grid1D,Grid1D,Grid1D)>{
        let slen = len as i64;
        let max_offs=self.offsets.iter().max().unwrap();
        let min_offs=self.offsets.iter().min().unwrap();
        None
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
