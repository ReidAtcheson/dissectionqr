use ndarray::{Array,Array2};
use ndarray::ShapeBuilder;
// LAPACK QR factorization routines
use lapack::{sgeqrf,dgeqrf,cgeqrf,zgeqrf};
// LAPACK triangular solve routines
use lapack::{strtrs,dtrtrs,ctrtrs,ztrtrs};
// LAPACK householder reflector routines
use lapack::{sormqr,dormqr,cunmqr,zunmqr};


use num_traits::{Num,Float,Zero};
use num_complex::Complex;


type C32=Complex<f32>;
type C64=Complex<f64>;

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


pub trait Lapack<F>{
    fn xgeqrf(m : i32, n : i32, a : &mut Array2<F>,lda : i32, tau : &mut Array2<F>,work : &mut Array2<F>,lwork : i32,info : &mut i32)->();
    fn xtrtrs(uplo : u8,trans : u8,diag : u8,n : i32,nrhs : i32,a : &mut Array2<F>,lda : i32,b : &mut Array2<F>,ldb : i32,info : &mut i32) -> ();
    fn xmqr(side : u8, trans : u8,m : i32,n : i32,k : i32,a : &mut Array2<F>,lda : i32,
        tau : &mut Array2<F>,c : &mut Array2<F>,ldc : i32,work : &mut Array2<F>,lwork : i32,info : &mut i32)->();
}
impl Lapack<f32> for f32{
    fn xgeqrf(m : i32, n : i32, a : &mut Array2<f32>,lda : i32, tau : &mut Array2<f32>,work : &mut Array2<f32>,lwork : i32,info : &mut i32)->(){
        let xp = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let yp = tau.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let zp = work.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            sgeqrf(m,n,xp,lda,yp,zp,lwork,info) 
        }
    }
    fn xtrtrs(uplo : u8,trans : u8,diag : u8,n : i32,nrhs : i32,a : &mut Array2<f32>,lda : i32,b : &mut Array2<f32>,ldb : i32,info : &mut i32) -> (){
        let ap = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let bp = b.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            strtrs(uplo,trans,diag,n,nrhs,ap,lda,bp,ldb,info) 
        }
    }
    fn xmqr(side : u8, trans : u8,m : i32,n : i32,k : i32,a : &mut Array2<f32>,lda : i32,
        tau : &mut Array2<f32>,c : &mut Array2<f32>,ldc : i32,work : &mut Array2<f32>,lwork : i32,info : &mut i32)->(){
        let ap = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let tp = tau.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let cp = c.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let wp = work.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            sormqr(side,trans,m,n,k,ap,lda,tp,cp,ldc,wp,lwork,info) 
        }
    }
}

impl Lapack<Complex<f32>> for Complex<f32>{
    fn xgeqrf(m : i32, n : i32, a : &mut Array2<Complex<f32>>,lda : i32, tau : &mut Array2<Complex<f32>>,work : &mut Array2<Complex<f32>>,lwork : i32,info : &mut i32)->(){
        let xp = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let yp = tau.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let zp = work.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            cgeqrf(m,n,xp,lda,yp,zp,lwork,info) 
        }
    }
    fn xtrtrs(uplo : u8,trans : u8,diag : u8,n : i32,nrhs : i32,a : &mut Array2<Complex<f32>>,lda : i32,b : &mut Array2<Complex<f32>>,ldb : i32,info : &mut i32) -> (){
        let ctrans= if trans==b'T'{b'C'} else {b'N'};
        let ap = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let bp = b.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            ctrtrs(uplo,ctrans,diag,n,nrhs,ap,lda,bp,ldb,info) 
        }
    }
    fn xmqr(side : u8, trans : u8,m : i32,n : i32,k : i32,a : &mut Array2<Complex<f32>>,lda : i32,
        tau : &mut Array2<Complex<f32>>,c : &mut Array2<Complex<f32>>,ldc : i32,work : &mut Array2<Complex<f32>>,lwork : i32,info : &mut i32)->(){
        let ctrans= if trans==b'T'{b'C'} else {b'N'};
        let ap = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let tp = tau.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let cp = c.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let wp = work.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            cunmqr(side,ctrans,m,n,k,ap,lda,tp,cp,ldc,wp,lwork,info) 
        }
    }

}


impl Lapack<f64> for f64{
    fn xgeqrf(m : i32, n : i32, a : &mut Array2<f64>,lda : i32, tau : &mut Array2<f64>,work : &mut Array2<f64>,lwork : i32,info : &mut i32)->(){
        let xp = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let yp = tau.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let zp = work.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            dgeqrf(m,n,xp,lda,yp,zp,lwork,info) 
        }
    }
    fn xtrtrs(uplo : u8,trans : u8,diag : u8,n : i32,nrhs : i32,a : &mut Array2<f64>,lda : i32,b : &mut Array2<f64>,ldb : i32,info : &mut i32) -> (){
        let ap = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let bp = b.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            dtrtrs(uplo,trans,diag,n,nrhs,ap,lda,bp,ldb,info) 
        }
    }
    fn xmqr(side : u8, trans : u8,m : i32,n : i32,k : i32,a : &mut Array2<f64>,lda : i32,
        tau : &mut Array2<f64>,c : &mut Array2<f64>,ldc : i32,work : &mut Array2<f64>,lwork : i32,info : &mut i32)->(){
        let ap = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let tp = tau.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let cp = c.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let wp = work.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            dormqr(side,trans,m,n,k,ap,lda,tp,cp,ldc,wp,lwork,info) 
        }
    }
}

impl Lapack<Complex<f64>> for Complex<f64>{
    fn xgeqrf(m : i32, n : i32, a : &mut Array2<Complex<f64>>,lda : i32, tau : &mut Array2<Complex<f64>>,work : &mut Array2<Complex<f64>>,lwork : i32,info : &mut i32)->(){
        let xp = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let yp = tau.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let zp = work.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            zgeqrf(m,n,xp,lda,yp,zp,lwork,info) 
        }
    }
    fn xtrtrs(uplo : u8,trans : u8,diag : u8,n : i32,nrhs : i32,a : &mut Array2<Complex<f64>>,lda : i32,b : &mut Array2<Complex<f64>>,ldb : i32,info : &mut i32) -> (){
        let ctrans= if trans==b'T'{b'C'} else {b'N'};
        let ap = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let bp = b.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            ztrtrs(uplo,ctrans,diag,n,nrhs,ap,lda,bp,ldb,info) 
        }
    }
    fn xmqr(side : u8, trans : u8,m : i32,n : i32,k : i32,a : &mut Array2<Complex<f64>>,lda : i32,
        tau : &mut Array2<Complex<f64>>,c : &mut Array2<Complex<f64>>,ldc : i32,work : &mut Array2<Complex<f64>>,lwork : i32,info : &mut i32)->(){
        let ctrans= if trans==b'T'{b'C'} else {b'N'};
        let ap = a.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let tp = tau.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let cp = c.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        let wp = work.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
        unsafe{
            zunmqr(side,ctrans,m,n,k,ap,lda,tp,cp,ldc,wp,lwork,info) 
        }
    }
}
/// A simple class wrapping LAPACK's xGEQRF and presenting a 
/// nicer interface for rust code.
pub struct QRFact<F>{
    m : i32,
    n : i32,
    qr : Array2<F>,
    tau : Array2<F>,
    work : Array2<F>,
    lwork : i32,
}





impl <F>  QRFact<F>
where
F : Clone + Zero + Lapack<F>{
    /// Take ownership of matrix `a` and do an in-place QR factorization
    pub fn new<'b>( a : Array2<F> )->Self{
        check_if_fortran_style(&a).expect("Matrix input to QRFact was not column-major ordered");
        let shape = a.shape();
        let m = shape[0] as i32;
        let n = shape[1] as i32;
        let lda = m as i32;
        let ntau = std::cmp::min(m,n) as i32;
        let nb=std::cmp::min(32,n);
        let lwork = nb * n as i32;

        let t : Array2<F> = Array::zeros((ntau as usize,1).f());
        let w : Array2<F> = Array::zeros((lwork as usize,1).f());
        {
            let mut qr  = a;
            let mut tau =t; 
            let mut work=w;
            let mut info=0 as i32;
            F::xgeqrf(m,n,&mut qr,lda,&mut tau,&mut work,lwork,&mut info);
            check_lapack_info(info).expect(&format!("xgeqrf failed with info={}",info)[..]);

            QRFact { m : m as i32, n : n as i32, qr : qr, tau : tau, work : work, lwork : lwork}
        }
    }
    /// Solve Rx=b in-place with the array b.
    pub fn solve_r(&mut self,b : Array2<F>) -> Array2<F> {
        check_if_fortran_style(&b).expect("Matrix B in RX=B not column-major ordered");
        let n = self.n;
        let nrhs = b.shape()[1] as i32;
        let k = std::cmp::min(self.m,self.n);
        {
            let shape = b.shape();
            if shape[0] as i32 != k as i32{
                panic!(format!("Error: R in RX=B has shape ({},{}) but B has shape ({},{})",k,n,shape[0],shape[1]));
            }
        }
        let lda=self.m;
        let ldb={
            let shape = b.shape();
            shape[0]
        } as i32;

        {
            let mut info = 0 as i32;
            let mut bb = b;
            F::xtrtrs(b'U',b'N',b'N',n,nrhs,&mut self.qr,lda,&mut bb,ldb,&mut info);
            check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
            bb
        }
    }

    /// Solve R^T x=b in-place with the array b.
    pub fn solve_rt(&mut self,b : Array2<F>) -> Array2<F> {
        check_if_fortran_style(&b).expect("Matrix B in RX=B not column-major ordered");
        let m = self.m;
        let n = self.n;
        let nrhs = {
            let shape=b.shape();
            shape[1]
        } as i32;
        let k = std::cmp::min(m,n);
        {
            let shape = b.shape();
            if shape[0] as i32 != n {
                panic!(format!("Error: R in R^T X=B has shape ({},{}) but B has shape ({},{})",k,n,shape[0],shape[1]));
            }
        }
        let lda=self.m;
        let ldb={
            let shape=b.shape();
            shape[0]
        } as i32;
        {
            let mut info = 0;
            let mut bb =b;
            F::xtrtrs(b'U',b'T',b'N',n,nrhs,&mut self.qr,lda,&mut bb,ldb,&mut info);
            check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
            bb
        }
    }

    /// Multiply Q^T * C in-place in C.
    pub fn mul_left_qt(&mut self,c : Array2<F>) -> Array2<F> {
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
        let k = std::cmp::min(m,n) as i32;
        let lda = m as i32;
        let ldc = c.shape()[0] as i32;
        let lwork=self.lwork;
        {
            let mut info = 0 as i32;
            let mut cc = c;
            F::xmqr(b'L',b'T',m,n,k,&mut self.qr,lda,&mut self.tau,&mut cc,ldc,&mut self.work,lwork,&mut info);
            check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
            cc
        }
    }

    /// Multiply Q * C in-place in C.
    pub fn mul_left_q(&mut self,c :Array2<F>) -> Array2<F> {
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
        let k = std::cmp::min(self.m,self.n) as i32;
        let lda = self.m as i32;
        let ldc = c.shape()[0] as i32;
        let lwork=self.lwork;
        {
            let mut info = 0 as i32;
            let mut cc  = c;
            F::xmqr(b'L',b'N',m,n,k,&mut self.qr,lda,&mut self.tau,&mut cc,ldc,&mut self.work,lwork,&mut info);
            check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
            cc
        }
    }
}





#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use num_traits::One;
    use num_traits::Num;
    use num_traits::Float;
    use num::NumCast;
    use num_complex::Complex;
    use num_complex::Complex32;
    use num_complex::Complex64;



    fn test_qr_correct_normal<F>() -> () 
    where 
    F : Float + NumCast + std::fmt::Display + Lapack<F>{
        let eps=F::epsilon();
        let scaling : F = NumCast::from(50 as i64).unwrap();
        let tol=scaling*eps;
        let m=50;
        let mut a : Array2<F>  = Array::zeros((m,m).f());
        for ((i,j),v) in a.indexed_iter_mut(){
            if i != j{
                let ifl : F = NumCast::from(i).unwrap();
                let jfl : F = NumCast::from(j).unwrap();
                let alpha : F= NumCast::from(3).unwrap();
                *v=(ifl+alpha*jfl).sin();
            }
            else{
                *v=NumCast::from(15).unwrap();
            }
        }
        let mut qr = QRFact::<F>::new(a.clone());
        a=qr.mul_left_qt(a);
        a=qr.solve_r(a);
        //Now A should be the identity
        for ((i,j),v) in a.indexed_iter(){
            if i==j{
                print!("\nv-one  = {}\n",(*v-F::one()).abs());
                assert!( (*v-F::one()).abs()<tol );
            }
            else{
                assert!( v.abs()<tol );
            }
        }
    }

    fn test_qr_correct_normal_complex<F>() -> () 
    where 
    F : Num + NumCast + Float + Clone + std::fmt::Display,
    Complex<F> : Num + NumCast + std::fmt::Display + Lapack<Complex<F>>{
        let eps=F::epsilon();
        let scaling : F = NumCast::from(50 as i64).unwrap();
        let tol=scaling*eps;
        let m=50;
        let mut a : Array2<Complex<F>>  = Array::zeros((m,m).f());
        for ((i,j),v) in a.indexed_iter_mut(){
            if i != j{
                let ifl : F = NumCast::from(i).unwrap();
                let jfl : F = NumCast::from(j).unwrap();
                let alpha : F= NumCast::from(3).unwrap();
                *v=Complex::<F>::new((ifl+alpha*jfl).sin(),(alpha*ifl+jfl).cos());
            }
            else{
                *v=Complex::<F>::new(NumCast::from(15).unwrap(),NumCast::from(-15).unwrap());
            }
        }
        let mut qr = QRFact::<Complex<F>>::new(a.clone());
        a=qr.mul_left_qt(a);
        a=qr.solve_r(a);
        //Now A should be the identity
        for ((i,j),v) in a.indexed_iter(){
            if i==j{
                print!("\nv-one  = {}\n",(*v-Complex::<F>::one()).norm());
                assert!( (*v-Complex::<F>::one()).norm()<tol );
            }
            else{
                assert!( v.norm()<tol );
            }
        }
    }




    #[test]
    fn test_qr_correct_normal_f32(){
        test_qr_correct_normal::<f32>();
    }
    #[test]
    fn test_qr_correct_normal_f64(){
        test_qr_correct_normal::<f64>();
    }
    #[test]
    fn test_qr_correct_normal_c32(){
        test_qr_correct_normal_complex::<f32>();
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
        a=qr.solve_rt(a);
        a=qr.mul_left_q(a);
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
        id=qr.mul_left_q(id);
        id=qr.mul_left_qt(id);
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
        qr.solve_r(b);
    }

    #[test]
    #[should_panic]
    fn test_qr_rsolve_left_mul_qt_dims_match_f32(){
        let a : Array2<f32>  = Array::from_shape_vec((3,3),vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let mut qr = QRFact::<f32>::new(a);
        //This is incorrect shape for b, this should fail.
        let mut b : Array2<f32>  = Array::from_shape_vec((2,3),vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
        qr.mul_left_qt(b);
    }


    /**
     *
     * FLOAT64 TESTS
     *
     */

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
        a=qr.solve_rt(a);
        a=qr.mul_left_q(a);
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
        id=qr.mul_left_q(id);
        id=qr.mul_left_qt(id);

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
        b=qr.solve_r(b);
    }

    #[test]
    #[should_panic]
    fn test_qr_rsolve_left_mul_qt_dims_match_f64(){
        let a : Array2<f64>  = Array::from_shape_vec((3,3),vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let mut qr = QRFact::<f64>::new(a);
        //This is incorrect shape for b, this should fail.
        let mut b : Array2<f64>  = Array::from_shape_vec((2,3),vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
        b=qr.mul_left_qt(b);
    }


}



