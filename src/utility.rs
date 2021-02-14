use ndarray::{Array,Array2};
use ndarray::ShapeBuilder;
// LAPACK QR factorization routines
use lapack::{sgeqrf,dgeqrf,cgeqrf,zgeqrf};
// LAPACK triangular solve routines
use lapack::{strtrs,dtrtrs,ctrtrs,ztrtrs};
// LAPACK householder reflector routines
use lapack::{sormqr,dormqr,cunmqr,zunmqr};
use num_traits::{Float,Zero};


//I want to add these but they do not implement any trait in common with
//floats
use lapack::c32;
use lapack::c64;
type C32=c32;
type C64=c64;

fn check_if_fortran_style<F>(a : &Array2<F>) -> Option<()>{
    let strides=a.strides();
    let shape=a.shape();
    let m=shape[0];
    let n=shape[0];
    if !(strides[0]==1 && strides[1]==m as isize){
        None
    }
    else{
        Some(())
    }
}

fn check_lapack_info(info : i32) -> Option<()>{
    if info!=0{
        None
    }
    else{
        Some(())
    }
}

enum NumSlice<'a>{
    Float32(&'a mut [f32]),
    Float64(&'a mut [f64]),
    Complex32(&'a mut [C32]),
    Complex64(&'a mut [C64])
}

impl From<&mut [f32]> for NumSlice<'a>{
    fn from(x : &'a mut [f32]) -> Self{
        NumSlice::Float32(x)
    }
}

// Dynamic type that can hold one of 4 very similar types
enum NumNDArray{
    Float32(Array2<f32>),
    Float64(Array2<f64>),
    Complex32(Array2<C32>),
    Complex64(Array2<C64>)
}

//Implement "From" conversions for the four types
//This automatically implements "Into" as well
impl From<Array2<f32>> for NumNDArray{
    fn from(arr : Array2<f32>) -> Self {
        NumNDArray::Float32(arr)
    }
}
impl From<Array2<f64>> for NumNDArray{
    fn from(arr : Array2<f64>) -> Self {
        NumNDArray::Float64(arr)
    }
}
impl From<Array2<C32>> for NumNDArray{
    fn from(arr : Array2<C32>) -> Self {
        NumNDArray::Complex32(arr)
    }
}
impl From<Array2<C64>> for NumNDArray{
    fn from(arr : Array2<C64>) -> Self {
        NumNDArray::Complex64(arr)
    }
}
impl From<NumNDArray> for Array2<f32> {
    fn from(arr : NumNDArray) -> Self {
        match arr{
            NumNDArray::Float32(arr) => arr,
            _ => panic!("Mismatched conversion from NumNDArray to Array2")
        }
    }
}
impl From<NumNDArray> for Array2<f64> {
    fn from(arr : NumNDArray) -> Self {
        match arr{
            NumNDArray::Float64(arr) => arr,
            _ => panic!("Mismatched conversion from NumNDArray to Array2")
        }
    }
}

impl From<NumNDArray> for Array2<C32> {
    fn from(arr : NumNDArray) -> Self {
        match arr{
            NumNDArray::Complex32(arr) => arr,
            _ => panic!("Mismatched conversion from NumNDArray to Array2")
        }
    }
}
impl From<NumNDArray> for Array2<C64> {
    fn from(arr : NumNDArray) -> Self {
        match arr{
            NumNDArray::Complex64(arr) => arr,
            _ => panic!("Mismatched conversion from NumNDArray to Array2")
        }
    }
}






fn xgeqrf(m : i32, n : i32, a : NumSlice,lda : i32, tau : NumSlice,work : NumSlice,lwork : i32,info : &mut i32) -> () {
    use crate::utility::NumSlice::Float32;
    use crate::utility::NumSlice::Float64;
    use crate::utility::NumSlice::Complex32;
    use crate::utility::NumSlice::Complex64;
    match (a,tau,work){        
        (Float32(x),Float32(y),Float32(z)) => unsafe { sgeqrf(m,n,x,lda,y,z,lwork,info) },
        (Float64(x),Float64(y),Float64(z)) => unsafe { dgeqrf(m,n,x,lda,y,z,lwork,info) },
        (Complex32(x),Complex32(y),Complex32(z)) => unsafe { cgeqrf(m,n,x,lda,y,z,lwork,info) },
        (Complex64(x),Complex64(y),Complex64(z)) => unsafe { zgeqrf(m,n,x,lda,y,z,lwork,info) },
        _ => panic!("Type mismatch in xgeqrf")
    };
}
fn xtrtrs(uplo : u8,trans : u8,diag : u8,n : i32,nrhs : i32,a : NumSlice,lda : i32,b : NumSlice,ldb : i32,info : &mut i32) -> () {
    use crate::utility::NumSlice::Float32;
    use crate::utility::NumSlice::Float64;
    use crate::utility::NumSlice::Complex32;
    use crate::utility::NumSlice::Complex64;
    let ctrans= if trans==b'T'{b'C'} else {b'N'};
    match (a,b){
        (Float32(a),Float32(b)) => unsafe { strtrs(uplo,trans,diag,n,nrhs,a,lda,b,ldb,info) },
        (Float64(a),Float64(b)) => unsafe { dtrtrs(uplo,trans,diag,n,nrhs,a,lda,b,ldb,info) },
        (Complex32(a),Complex32(b)) => unsafe { ctrtrs(uplo,ctrans,diag,n,nrhs,a,lda,b,ldb,info) },
        (Complex64(a),Complex64(b)) => unsafe { ztrtrs(uplo,ctrans,diag,n,nrhs,a,lda,b,ldb,info) },
        _ => panic!("Type mismatch in xtrtrs")
    }
}

fn xmqr(side : u8, trans : u8,m : i32,n : i32,k : i32,a : NumSlice,lda : i32,tau : NumSlice,c : NumSlice,ldc : i32,work : NumSlice,lwork : i32,info : &mut i32)->(){
    use crate::utility::NumSlice::Float32;
    use crate::utility::NumSlice::Float64;
    use crate::utility::NumSlice::Complex32;
    use crate::utility::NumSlice::Complex64;
    let ctrans= if trans==b'T'{b'C'} else {b'N'};
    match (a,tau,c,work){
        (Float32(a),Float32(tau),Float32(c),Float32(work)) => unsafe { sormqr(side,trans,m,n,k,a,lda,tau,c,ldc,work,lwork,info) },
        (Float64(a),Float64(tau),Float64(c),Float64(work)) => unsafe { dormqr(side,trans,m,n,k,a,lda,tau,c,ldc,work,lwork,info) },
        (Complex32(a),Complex32(tau),Complex32(c),Complex32(work)) => unsafe { cunmqr(side,ctrans,m,n,k,a,lda,tau,c,ldc,work,lwork,info) },
        (Complex64(a),Complex64(tau),Complex64(c),Complex64(work)) => unsafe { zunmqr(side,ctrans,m,n,k,a,lda,tau,c,ldc,work,lwork,info) },
        _ => panic!("Type mismatch in xmqr")
    }
}

impl NumNDArray{
    fn get_mutable_slice(&mut self) -> NumSlice {
        match self{
            NumNDArray::Float32(a) => NumSlice::Float32(a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix")),
            NumNDArray::Float64(a) => NumSlice::Float64(a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix")),
            NumNDArray::Complex32(a) => NumSlice::Complex32(a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix")),
            NumNDArray::Complex64(a) => NumSlice::Complex64(a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix"))
        }
    }
}



/// A simple class wrapping LAPACK's xGEQRF and presenting a 
/// nicer interface for rust code.
pub struct QRFact<F>{
    m : i32,
    n : i32,
    qr : NumNDArray,
    tau : Vec<F>,
    work : Vec<F>,
    lwork : i32,
}

impl <F> QRFact<F>
where 
    F : Clone + Zero,
    Array2<F>  : Into<NumNDArray> + From<NumNDArray> 
{
    /// Take ownership of matrix `a` and do an in-place QR factorization
    pub fn new( a : Array2<F> )->Self{

        let shape = a.shape();
        let m = shape[0] as i32;
        let n = shape[1] as i32;
        check_if_fortran_style(&a).expect("Matrix input to QRFact was not column-major ordered");
        let lda = m as i32;
        let ntau = std::cmp::min(m,n) as i32;
        let mut tau : Vec<F> = vec![ F::zero() ;ntau as usize];
        let nb=std::cmp::min(32,n);
        let lwork = nb * 2 as i32;
        let mut work : Vec<F> = vec![ F::zero() ;lwork as usize];
        let mut info=0 as i32;
        let qr : NumNDArray = {
            let mut qr : NumNDArray = a.into();
            let qrslice = qr.get_mutable_slice();
            xgeqrf(m,n,qrslice,lda,&mut tau,&mut work,lwork,&mut info);
            check_lapack_info(info).expect(&format!("sgeqrf failed with info={}",info)[..]);
            qr
        };
        QRFact { m : m as i32, n : n as i32, qr : qr, tau : tau, work : work, lwork : lwork,tmp : vec![]}
    }
    /// Solve Rx=b in-place with the array b.
    pub fn solve_r(&mut self,b : Array2<F>) -> Array2<F> {
        check_if_fortran_style(b).expect("Matrix B in RX=B not column-major ordered");
        let n = self.n;
        let nrhs = b.shape()[1] as i32;
        let k = std::cmp::min(self.qr.shape()[0],self.qr.shape()[0]);
        {
            let shape = b.shape();
            if shape[0] as i32 != k as i32{
                panic!(format!("Error: R in RX=B has shape ({},{}) but B has shape ({},{})",k,n,shape[0],shape[1]));
            }
        }
        let qrslice=self.qr.get_mutable_slice();
        let lda=self.m;
        let ldb={
            let shape = b.shape();
            shape[0]
        } as i32;

        {
            let mut x : NumNDArray = b.into();
            let bslice=x.get_mutable_slice();
            let mut info = 0 as i32;
            xtrtrs(b'U',b'N',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
            check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);        
            x.from()
        }
    }

    /// Solve R^T x=b in-place with the array b.
    pub fn solve_rt(&mut self,b : Array2<F>) -> Array2<F> {
        check_if_fortran_style(b).expect("Matrix B in RX=B not column-major ordered");
        let n = self.n;
        let nrhs = b.shape()[1] as i32;
        let k = std::cmp::min(self.qr.shape()[0],self.qr.shape()[0]);
        {
            let shape = b.shape();
            if shape[0] as i32 != k as i32{
                panic!(format!("Error: R in RX=B has shape ({},{}) but B has shape ({},{})",k,n,shape[0],shape[1]));
            }
        }
        let qrslice=self.qr.get_mutable_slice();
        let lda=self.m;
        let ldb={
            let shape = b.shape();
            shape[0]
        } as i32;

        {
            let mut x : NumNDArray = b.into();
            let bslice=x.get_mutable_slice();
            let mut info = 0 as i32;
            xtrtrs(b'U',b'T',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
            check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
            x.from()
        }
    }



    /// Multiply Q^T * C in-place in C.
    pub fn mul_left_qt(&mut self,c : Array2<F>) -> Array2<F> {
        check_if_fortran_style(c).expect("Matrix C in Q^T * C not column-major ordered");
        {
            let m = self.m;
            let shape=c.shape();
            if shape[0] as i32 != m {
                panic!(format!("Error: Q in Q^T * C has shape ({},{}) but C has shape ({},{})",m,m,shape[0],shape[1]));
            }
        }
        let m = self.m as i32;
        let n = c.shape()[1] as i32;
        let k = self.tau.len() as i32;
        let lda = self.qr.shape()[0] as i32;
        let ldc = c.shape()[0] as i32;
        {
            let mut x : NumNDArray = c.into();
            let qrslice=self.qr.get_mutable_slice();
            let cslice=x.get_mutable_slice();
            let lwork=self.lwork;
            let mut info = 0 as i32;
            xmqr(b'L',b'T',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
            check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
            x.from()
        }
    }

    /// Multiply Q * C in-place in C.
    pub fn mul_left_q(&mut self,c : Array2<F>) -> Array2<F> {
        check_if_fortran_style(&c).expect("Matrix C in Q^T * C not column-major ordered");
        {
            let m = self.m;
            let shape=c.shape();
            if shape[0] as i32 != m {
                panic!(format!("Error: Q in Q^T * C has shape ({},{}) but C has shape ({},{})",m,m,shape[0],shape[1]));
            }
        }
        let m = self.m as i32;
        let n = c.shape()[1] as i32;
        let k = self.tau.len() as i32;
        let lda = m as i32;
        let ldc = c.shape()[0] as i32;
        {
            let mut x : NumNDArray = c.into();
            let qrslice=self.qr.get_mutable_slice();
            let cslice=x.get_mutable_slice();
            let lwork=self.lwork;
            let mut info = 0 as i32;
            xmqr(b'L',b'N',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
            check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
            x.from()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use num_traits::Float;




    #[test]
    fn test_qr_correct_normal_f32(){
        type F=f32;
        let eps=F::epsilon();
        let tol=50.0*eps;
        let m=50;
        let mut a : Array2<F>  = Array::zeros((m,m).f());
        for ((i,j),v) in a.indexed_iter_mut(){
            if i != j{
                let ifl = i as F;
                let jfl = j as F;
                *v=(ifl+3.0*jfl).sin();
            }
            else{
                *v=15.0;
            }
        }
        let mut qr = QRFact::<F>::new(a.clone());
        qr.mul_left_qt(&mut a);
        qr.solve_r(&mut a);
        //Now A should be the identity
        for ((i,j),v) in a.indexed_iter(){
            if i==j{
                print!("\nv-one  = {}\n",(v-1.0).abs());
                assert!( (v-1.0).abs()<tol );
            }
            else{
                assert!( v.abs()<tol );
            }
        }
    }
    #[test]
    fn test_qr_correct_transpose_f32(){
        type F=f32;
        let eps=F::epsilon();
        let tol=50.0*eps;
        let m=50;
        let mut a : Array2<F>  = Array::zeros((m,m).f());
        let mut at : Array2<F>  = Array::zeros((m,m).f());
        for ((i,j),v) in a.indexed_iter_mut(){
            if i != j{
                let ifl = i as F;
                let jfl = j as F;
                *v=(ifl+3.0*jfl).sin();
            }
            else{
                *v=15.0;
            }
        }

        for ((i,j),v) in a.indexed_iter(){
            at[[j,i]]=*v
        }


        let mut qr = QRFact::<F>::new(at);
        qr.solve_rt(&mut a);
        qr.mul_left_q(&mut a);
        //Now A should be the identity
        for ((i,j),v) in a.indexed_iter(){
            if i==j{
                print!("\nv-one  = {}\n",(v-1.0).abs());
                assert!( (v-1.0).abs()<tol );
            }
            else{
                assert!( v.abs()<tol );
            }
        }
    }

    #[test]
    fn test_qr_correct_orthogonal_f32(){
        type F=f32;
        let eps=F::epsilon();
        let tol=50.0*eps;
        let m=50;
        let mut a : Array2<F>  = Array::zeros((m,m).f());
        let mut id : Array2<F> = Array::zeros((m,m).f());
        for ((i,j),v) in a.indexed_iter_mut(){
            if i != j{
                let ifl = i as F;
                let jfl = j as F;
                *v=(ifl+3.0*jfl).sin();
            }
            else{
                *v=15.0;
            }
        }
        for ((i,j),v) in id.indexed_iter_mut(){
            if i==j{
                *v=1.0;
            }
        }
        let mut qr = QRFact::<F>::new(a.clone());
        qr.mul_left_q(&mut id);
        qr.mul_left_qt(&mut id);
        for ((i,j),v) in id.indexed_iter(){
            if i==j{
                print!("\nv-one  = {}\n",(v-1.0).abs());
                assert!( (v-1.0).abs()<tol );
            }
            else{
                assert!( v.abs()<tol );
            }
        }
    }


    #[test]
    #[should_panic]
    fn test_qr_c_major_input_f32(){
        let a : Array2<f32>  = Array::from_shape_vec((3,3),vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let mut qr = QRFact::<f32>::new(a);
    }
    #[test]
    #[should_panic]
    fn test_qr_rsolve_dims_match_f32(){
        let a : Array2<f32>  = Array::from_shape_vec((3,3),vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let mut qr = QRFact::<f32>::new(a);
        //This is incorrect shape for b, it should fail.
        let mut b : Array2<f32>  = Array::from_shape_vec((2,3),vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
        qr.solve_r(&mut b);
    }

    #[test]
    #[should_panic]
    fn test_qr_rsolve_left_mul_qt_dims_match_f32(){
        let a : Array2<f32>  = Array::from_shape_vec((3,3),vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let mut qr = QRFact::<f32>::new(a);
        //This is incorrect shape for b, this should fail.
        let mut b : Array2<f32>  = Array::from_shape_vec((2,3),vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
        qr.mul_left_qt(&mut b);
    }


    /**
     *
     * FLOAT64 TESTS
     *
     */

    #[test]
    fn test_qr_correct_normal_f64(){
        type F=f64;
        let eps=F::epsilon();
        let tol=50.0*eps;
        let m=50;
        let mut a : Array2<F>  = Array::zeros((m,m).f());
        for ((i,j),v) in a.indexed_iter_mut(){
            if i != j{
                let ifl = i as F;
                let jfl = j as F;
                *v=(ifl+3.0*jfl).sin();
            }
            else{
                *v=15.0;
            }
        }
        let mut qr = QRFact::<F>::new(a.clone());
        qr.mul_left_qt(&mut a);
        qr.solve_r(&mut a);
        //Now A should be the identity
        for ((i,j),v) in a.indexed_iter(){
            if i==j{
                print!("\nv-one  = {}\n",(v-1.0).abs());
                assert!( (v-1.0).abs()<tol );
            }
            else{
                assert!( v.abs()<tol );
            }
        }
    }
    #[test]
    fn test_qr_correct_transpose_f64(){
        type F=f64;
        let eps=F::epsilon();
        let tol=50.0*eps;
        let m=50;
        let mut a : Array2<F>  = Array::zeros((m,m).f());
        let mut at : Array2<F>  = Array::zeros((m,m).f());
        for ((i,j),v) in a.indexed_iter_mut(){
            if i != j{
                let ifl = i as F;
                let jfl = j as F;
                *v=(ifl+3.0*jfl).sin();
            }
            else{
                *v=15.0;
            }
        }

        for ((i,j),v) in a.indexed_iter(){
            at[[j,i]]=*v
        }


        let mut qr = QRFact::<F>::new(at);
        qr.solve_rt(&mut a);
        qr.mul_left_q(&mut a);
        //Now A should be the identity
        for ((i,j),v) in a.indexed_iter(){
            if i==j{
                print!("\nv-one  = {}\n",(v-1.0).abs());
                assert!( (v-1.0).abs()<tol );
            }
            else{
                assert!( v.abs()<tol );
            }
        }
    }

    #[test]
    fn test_qr_correct_orthogonal_f64(){
        type F=f64;
        let eps=F::epsilon();
        let tol=50.0*eps;
        let m=50;
        let mut a : Array2<F>  = Array::zeros((m,m).f());
        let mut id : Array2<F> = Array::zeros((m,m).f());
        for ((i,j),v) in a.indexed_iter_mut(){
            if i != j{
                let ifl = i as F;
                let jfl = j as F;
                *v=(ifl+3.0*jfl).sin();
            }
            else{
                *v=15.0;
            }
        }
        for ((i,j),v) in id.indexed_iter_mut(){
            if i==j{
                *v=1.0;
            }
        }
        let mut qr = QRFact::<F>::new(a.clone());
        qr.mul_left_q(&mut id);
        qr.mul_left_qt(&mut id);
        for ((i,j),v) in id.indexed_iter(){
            if i==j{
                print!("\nv-one  = {}\n",(v-1.0).abs());
                assert!( (v-1.0).abs()<tol );
            }
            else{
                assert!( v.abs()<tol );
            }
        }
    }


    #[test]
    #[should_panic]
    fn test_qr_c_major_input_f64(){
        let a : Array2<f64>  = Array::from_shape_vec((3,3),vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let mut qr = QRFact::<f64>::new(a);
    }
    #[test]
    #[should_panic]
    fn test_qr_rsolve_dims_match_f64(){
        let a : Array2<f64>  = Array::from_shape_vec((3,3),vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let mut qr = QRFact::<f64>::new(a);
        //This is incorrect shape for b, it should fail.
        let mut b : Array2<f64>  = Array::from_shape_vec((2,3),vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
        qr.solve_r(&mut b);
    }

    #[test]
    #[should_panic]
    fn test_qr_rsolve_left_mul_qt_dims_match_f64(){
        let a : Array2<f64>  = Array::from_shape_vec((3,3),vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let mut qr = QRFact::<f64>::new(a);
        //This is incorrect shape for b, this should fail.
        let mut b : Array2<f64>  = Array::from_shape_vec((2,3),vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
        qr.mul_left_qt(&mut b);
    }


}



