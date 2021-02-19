
use crate::grid::*;
use crate::sparse::*;
use crate::dtree::*;
use crate::numeric::*;
use ndarray::prelude::*;

use ndarray::prelude::Ix2;
use ndarray::{Array,Array2};
use num_traits::Zero;

#[derive(Clone,Debug)]
pub struct SimpleStencil1D{
    cs : [f64;3],
    s : Grid1D,
    g : Stencil1D
}


impl SparseMatrix<f64,Grid1D,Stencil1D> for SimpleStencil1D{
    fn nrows(&self) -> usize{
        self.s.nnodes()
    }
    fn ncols(&self) -> usize{
        self.s.nnodes()
    }
    fn assemble(&self,rows : &Grid1D,cols : &Grid1D) -> Array2<f64>{
        let nrows=rows.nnodes();
        let ncols=cols.nnodes();
        let offs=&self.g.offsets;
        let mut a = Array2::<f64>::zeros((nrows,ncols).f());

        for gr in (rows.beg..rows.end){
            for i in 0..3{
                let off=offs[i];
                let lr = gr - rows.beg as i64;
                let lc = gr + off - cols.beg as i64;
                let gc = gr + off;
                if gc>=cols.beg  && gc<cols.end{
                    a[[lr as usize,lc as usize]]=self.cs[i as usize];
                }
            }
        }
        a
    }

    fn eval(&self,x : &Array2<f64>,y : &mut Array2<f64>) -> (){
        let sx = x.shape();
        let sy = y.shape();
        if sx != sy{
            panic!("Input and output array shapes do not match");
        }
        if self.s.nnodes() != sx[0]{
            panic!("Shape mismatch between sparse matrix and input array");
        }

        let offs=&self.g.offsets;
        for ((r,j),v) in y.indexed_iter_mut(){
            *v=0.0;
            for i in 0..3{
                let off = offs[i];
                let c=((j as i64)+off);
                let beg=self.s.beg as i64;
                let end=self.s.end as i64;
                if c>= beg && c<end{
                    *v += self.cs[i as usize] * x[[r as usize,c as usize]];
                } 
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One,Zero};

    fn identity_matrix<F : Clone+Zero+One>(nrows : usize)->Array2<F>{
        let mut out = Array2::<F>::zeros((nrows,nrows).f());
        for ((r,c),v) in out.indexed_iter_mut(){
            if r==c{
                *v=F::one();
            }
        }
        out
    }

    #[test]
    fn stencil1d_factorize(){

        let nrows=1000;
        let ncols=1000;
        //Set up problem with a nodeset and graph over that nodeset
        let s = Grid1D{beg : 0, end : nrows.clone()};
        let g = Stencil1D{ offsets : vec![-1,0,1] };
        //Combine into simple structure for simplicity - this implements "sparse matrix"
        let m = SimpleStencil1D{ cs : [1.0,-2.0,1.0],s : s, g : g };

        //Maximum number of nodes we allow a leaf to have in the nested dissection tree.
        //Smaller maxnodes results in more leafs in the tree and therefore more
        //potential for parallelism and less fill-in, but the level-3 BLAS
        //for dense QR factorizations becomes less efficient when this is
        //too small as well.
        let maxnodes=50;
        //How big do we want our separators? Here we require the separators to 
        //put a space so that no paths of length 2 exist between the two partitions
        //it separates
        let pathlen=2;
        //Perform the nested dissection and compute its tree
        //This is similar to symbolic factorization in other packages
        //but a little simpler because I don't compute a quotient graph.
        let dtree = nested_dissection(&m.s,&m.g,maxnodes,pathlen);
        //With symbolic factorization done we can now allocate all the
        //memory to hold fill-in and compute the resulting QR factorization
        let mut fact = assemble(dtree, m.clone());

        //Assemble the full matrix for the operator corresponding to 
        //the `SimpleStencil1D`
        let id = identity_matrix::<f64>(nrows as usize);
        let mut full = Array2::<f64>::zeros( (nrows as usize,ncols as usize).f() );
        m.eval(&id,&mut full);
        //Now that we have the full matrix representation
        //we can apply the factorization solve method on it
        //and the result should be equal to identity matrix
        //up to numerical roundoff times conditioning of the matrix
        //(which in this case is very small)
        full=fact.solve(full);


        let tol=1e-14;
        for ((r,c),v) in full.indexed_iter(){
            if r==c{
                assert!( (*v-1.0).abs() < tol );
            }
            else{
                assert!( (*v).abs() < tol );
            }
        }
    }
}




