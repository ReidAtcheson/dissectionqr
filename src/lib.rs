

pub trait NodeSet {
    fn contains(&self,i : i64) -> bool;
    fn disjoint(&self,s : Self) -> bool;
    fn enumerate(&self) -> Vec<i64>;
    fn nnodes(&self) -> i64;
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
    fn nnodes(&self) -> i64 {
        self.end-self.beg
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    //Simple sanity check that checks that an enumerated nodeset contains all its nodes
    fn nodeset_enumerate_all_in<S : NodeSet>(s : S) -> bool{
        let nodes = s.enumerate();
        let all_in = nodes.iter().map(|x|{s.contains(*x)}).fold(false,|acc,x|{acc||x});
        all_in
    }

    #[test]
    fn grid1d_sanity() {
        let g = Grid1D{ beg : 0,end : 10};
        assert!( nodeset_enumerate_all_in(g) );
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
