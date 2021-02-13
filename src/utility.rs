use ndarray::prelude::Ix2;
use ndarray::{Array,Array2,arr2};
// LAPACK QR factorization routines
use lapack::{sgeqrf,dgeqrf,cgeqrf,zgeqrf};
// LAPACK triangular solve routines
use lapack::{strtrs,dtrtrs,ctrtrs,ztrtrs};
// LAPACK householder reflector routines
use lapack::{sormqr,dormqr,cunmqr,zunmqr};


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







/// A simple class wrapping LAPACK's xGEQRF and presenting a 
/// nicer interface for rust code.
pub struct QRFact<F>{
    m : i32,
    n : i32,
    qr : Array2<F>,
    tau : Vec<F>,
    work : Vec<F>,
    lwork : i32
}



impl QRFact<f32>{
    /// Take ownership of matrix `a` and do an in-place QR factorization
    pub fn new( a : Array2<f32> )->Self{
        type F=f32;
        let shape = a.shape();

        let m = shape[0] as i32;
        let n = shape[1] as i32;
        check_if_fortran_style(&a).expect("Matrix input to QRFact was not column-major ordered");
        let mut qr = a;
        let lda = m as i32;
        let ntau = std::cmp::min(m,n) as i32;
        let mut tau : Vec<F> = vec![0.0;ntau as usize];
        let nb=std::cmp::min(32,n);
        let lwork = nb * 2 as i32;
        let mut work : Vec<F> = vec![0.0;lwork as usize];
        let mut info=0 as i32;
        unsafe{
            let qrslice = qr.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
            sgeqrf(m,n,qrslice,lda,&mut tau,&mut work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sgeqrf failed with info={}",info)[..]);
        QRFact { m : m as i32, n : n as i32, qr : qr, tau : tau, work : work, lwork : lwork}
    }
    /// Solve Rx=b in-place with the array b.
    pub fn solve_r(&mut self,b : &mut Array2<f32>) -> () {
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
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let lda=self.m;
        let ldb={
            let shape = b.shape();
            shape[0]
        } as i32;

        let bslice=b.as_slice_memory_order_mut().expect("Memory for B in RX=B not contiguous");
        let mut info = 0 as i32;
        unsafe {
            strtrs(b'U',b'N',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
        }
        check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
    }

    /// Solve R^T x=b in-place with the array b.
    pub fn solve_rt(&mut self,b : &mut Array2<f32>) -> () {
        check_if_fortran_style(b).expect("Matrix B in RX=B not column-major ordered");
        let n = self.n;
        let nrhs = {
            let shape=b.shape();
            shape[1]
        } as i32;
        let k = std::cmp::min(self.qr.shape()[0],self.qr.shape()[0]);
        {
            let shape = b.shape();
            if shape[0] as i32 != n {
                panic!(format!("Error: R in R^T X=B has shape ({},{}) but B has shape ({},{})",k,n,shape[0],shape[1]));
            }
        }
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let lda=self.m;
        let ldb={
            let shape=b.shape();
            shape[0]
        } as i32;
        let bslice=b.as_slice_memory_order_mut().expect("Memory for B in RX=B not contiguous");
        let mut info = 0;
        unsafe {
            strtrs(b'U',b'T',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
        }
        check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
    }

    /// Multiply Q^T * C in-place in C.
    pub fn mul_left_qt(&mut self,c : &mut Array2<f32>) -> () {
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
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for Q in Q^T * C not contiguous");
        let cslice=c.as_slice_memory_order_mut().expect("Memory for C in Q^T * C not contiguous");
        let lwork=self.lwork;
        let mut info = 0 as i32;
        unsafe{ 
            sormqr(b'L',b'T',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
    }

    /// Multiply Q * C in-place in C.
    pub fn mul_left_q(&mut self,c : &mut Array2<f32>) -> () {
        check_if_fortran_style(c).expect("Matrix C in Q^T * C not column-major ordered");
        {
            let m = self.m;
            let shape=c.shape();
            if shape[0] as i32 != m {
                panic!(format!("Error: Q in Q * C has shape ({},{}) but C has shape ({},{})",m,m,shape[0],shape[1]));
            }
        }
        let m = self.m as i32;
        let n = c.shape()[1] as i32;
        let k = self.tau.len() as i32;
        let lda = self.qr.shape()[0] as i32;
        let ldc = c.shape()[0] as i32;
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for Q in Q * C not contiguous");
        let cslice=c.as_slice_memory_order_mut().expect("Memory for C in Q * C not contiguous");
        let lwork=self.lwork;
        let mut info = 0 as i32;
        unsafe{ 
            sormqr(b'L',b'N',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
    }


}

impl QRFact<f64>{
    /// Take ownership of matrix `a` and do an in-place QR factorization
    pub fn new( a : Array2<f64> )->Self{
        type F=f64;
        let shape = a.shape();

        let m = shape[0] as i32;
        let n = shape[1] as i32;
        check_if_fortran_style(&a).expect("Matrix input to QRFact was not column-major ordered");
        let mut qr = a;
        let lda = m as i32;
        let ntau = std::cmp::min(m,n) as i32;
        let mut tau : Vec<F> = vec![0.0;ntau as usize];
        let nb=std::cmp::min(32,n);
        let lwork = nb * 2 as i32;
        let mut work : Vec<F> = vec![0.0;lwork as usize];
        let mut info=0 as i32;
        unsafe{
            let qrslice = qr.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
            dgeqrf(m,n,qrslice,lda,&mut tau,&mut work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sgeqrf failed with info={}",info)[..]);
        QRFact { m : m as i32, n : n as i32, qr : qr, tau : tau, work : work, lwork : lwork}
    }
    /// Solve Rx=b in-place with the array b.
    pub fn solve_r(&mut self,b : &mut Array2<f64>) -> () {
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
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let lda=self.m;
        let ldb={
            let shape = b.shape();
            shape[0]
        } as i32;

        let bslice=b.as_slice_memory_order_mut().expect("Memory for B in RX=B not contiguous");
        let mut info = 0 as i32;
        unsafe {
            dtrtrs(b'U',b'N',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
        }
        check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
    }

    /// Solve R^T x=b in-place with the array b.
    pub fn solve_rt(&mut self,b : &mut Array2<f64>) -> () {
        check_if_fortran_style(b).expect("Matrix B in RX=B not column-major ordered");
        let n = self.n;
        let nrhs = {
            let shape=b.shape();
            shape[1]
        } as i32;
        let k = std::cmp::min(self.qr.shape()[0],self.qr.shape()[0]);
        {
            let shape = b.shape();
            if shape[0] as i32 != n {
                panic!(format!("Error: R in R^T X=B has shape ({},{}) but B has shape ({},{})",k,n,shape[0],shape[1]));
            }
        }
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let lda=self.m;
        let ldb={
            let shape=b.shape();
            shape[0]
        } as i32;
        let bslice=b.as_slice_memory_order_mut().expect("Memory for B in RX=B not contiguous");
        let mut info = 0;
        unsafe {
            dtrtrs(b'U',b'T',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
        }
        check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
    }

    /// Multiply Q^T * C in-place in C.
    pub fn mul_left_qt(&mut self,c : &mut Array2<f64>) -> () {
        check_if_fortran_style(c).expect("Matrix C in Q^T * C not column-major ordered");
        {
            let m = self.m;
            let shape=c.shape();
            if shape[0] as i32 != m {
                panic!(format!("Error: Q in Q^T has shape ({},{}) but C has shape ({},{})",m,m,shape[0],shape[1]));
            }
        }
        let m = self.m as i32;
        let n = c.shape()[1] as i32;
        let k = self.tau.len() as i32;
        let lda = self.qr.shape()[0] as i32;
        let ldc = c.shape()[0] as i32;
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let cslice=c.as_slice_memory_order_mut().expect("Memory for C in Q^T * C not contiguous");
        let lwork=self.lwork;
        let mut info = 0 as i32;
        unsafe{ 
            dormqr(b'L',b'T',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
    }

    /// Multiply Q * C in-place in C.
    pub fn mul_left_q(&mut self,c : &mut Array2<f64>) -> () {
        check_if_fortran_style(c).expect("Matrix C in Q^T * C not column-major ordered");
        {
            let m = self.m;
            let shape=c.shape();
            if shape[0] as i32 != m {
                panic!(format!("Error: Q in Q * C has shape ({},{}) but C has shape ({},{})",m,m,shape[0],shape[1]));
            }
        }
        let m = self.m as i32;
        let n = c.shape()[1] as i32;
        let k = self.tau.len() as i32;
        let lda = self.qr.shape()[0] as i32;
        let ldc = c.shape()[0] as i32;
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for Q in Q * C not contiguous");
        let cslice=c.as_slice_memory_order_mut().expect("Memory for C in Q * C not contiguous");
        let lwork=self.lwork;
        let mut info = 0 as i32;
        unsafe{ 
            dormqr(b'L',b'N',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
    }
}

impl QRFact<C32>{
    /// Take ownership of matrix `a` and do an in-place QR factorization
    pub fn new( a : Array2<C32> )->Self{
        type F=C32;
        let shape = a.shape();

        let m = shape[0] as i32;
        let n = shape[1] as i32;
        check_if_fortran_style(&a).expect("Matrix input to QRFact was not column-major ordered");
        let mut qr = a;
        let lda = m as i32;
        let ntau = std::cmp::min(m,n) as i32;
        let mut tau : Vec<F> = vec![C32::new(0.0,0.0);ntau as usize];
        let nb=std::cmp::min(32,n);
        let lwork = nb * 2 as i32;
        let mut work : Vec<F> = vec![C32::new(0.0,0.0);lwork as usize];
        let mut info=0 as i32;
        unsafe{
            let qrslice = qr.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
            cgeqrf(m,n,qrslice,lda,&mut tau,&mut work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sgeqrf failed with info={}",info)[..]);
        QRFact { m : m as i32, n : n as i32, qr : qr, tau : tau, work : work, lwork : lwork}
    }
    /// Solve Rx=b in-place with the array b.
    pub fn solve_r(&mut self,b : &mut Array2<C32>) -> () {
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
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let lda=self.m;
        let ldb={
            let shape = b.shape();
            shape[0]
        } as i32;

        let bslice=b.as_slice_memory_order_mut().expect("Memory for B in RX=B not contiguous");
        let mut info = 0 as i32;
        unsafe {
            ctrtrs(b'U',b'N',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
        }
        check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
    }

    /// Solve R^T x=b in-place with the array b.
    pub fn solve_rt(&mut self,b : &mut Array2<C32>) -> () {
        check_if_fortran_style(b).expect("Matrix B in RX=B not column-major ordered");
        let n = self.n;
        let nrhs = {
            let shape=b.shape();
            shape[1]
        } as i32;
        let k = std::cmp::min(self.qr.shape()[0],self.qr.shape()[0]);
        {
            let shape = b.shape();
            if shape[0] as i32 != n {
                panic!(format!("Error: R in R^T X=B has shape ({},{}) but B has shape ({},{})",k,n,shape[0],shape[1]));
            }
        }
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let lda=self.m;
        let ldb={
            let shape=b.shape();
            shape[0]
        } as i32;
        let bslice=b.as_slice_memory_order_mut().expect("Memory for B in RX=B not contiguous");
        let mut info = 0;
        unsafe {
            ctrtrs(b'U',b'T',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
        }
        check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
    }

    /// Multiply Q^T * C in-place in C.
    pub fn mul_left_qt(&mut self,c : &mut Array2<C32>) -> () {
        check_if_fortran_style(c).expect("Matrix C in Q^T * C not column-major ordered");
        {
            let m = self.m;
            let shape=c.shape();
            if shape[0] as i32 != m {
                panic!(format!("Error: Q in Q^T has shape ({},{}) but C has shape ({},{})",m,m,shape[0],shape[1]));
            }
        }
        let m = self.m as i32;
        let n = c.shape()[1] as i32;
        let k = self.tau.len() as i32;
        let lda = self.qr.shape()[0] as i32;
        let ldc = c.shape()[0] as i32;
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let cslice=c.as_slice_memory_order_mut().expect("Memory for C in Q^T * C not contiguous");
        let lwork=self.lwork;
        let mut info = 0 as i32;
        unsafe{ 
            cunmqr(b'L',b'T',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
    }

    /// Multiply Q * C in-place in C.
    pub fn mul_left_q(&mut self,c : &mut Array2<C32>) -> () {
        check_if_fortran_style(c).expect("Matrix C in Q^T * C not column-major ordered");
        {
            let m = self.m;
            let shape=c.shape();
            if shape[0] as i32 != m {
                panic!(format!("Error: Q in Q * C has shape ({},{}) but C has shape ({},{})",m,m,shape[0],shape[1]));
            }
        }
        let m = self.m as i32;
        let n = c.shape()[1] as i32;
        let k = self.tau.len() as i32;
        let lda = self.qr.shape()[0] as i32;
        let ldc = c.shape()[0] as i32;
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for Q in Q * C not contiguous");
        let cslice=c.as_slice_memory_order_mut().expect("Memory for C in Q * C not contiguous");
        let lwork=self.lwork;
        let mut info = 0 as i32;
        unsafe{ 
            cunmqr(b'L',b'N',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
    }
}

impl QRFact<C64>{
    /// Take ownership of matrix `a` and do an in-place QR factorization
    pub fn new( a : Array2<C64> )->Self{
        type F=C64;
        let shape = a.shape();

        let m = shape[0] as i32;
        let n = shape[1] as i32;
        check_if_fortran_style(&a).expect("Matrix input to QRFact was not column-major ordered");
        let mut qr = a;
        let lda = m as i32;
        let ntau = std::cmp::min(m,n) as i32;
        let mut tau : Vec<F> = vec![C64::new(0.0,0.0);ntau as usize];
        let nb=std::cmp::min(32,n);
        let lwork = nb * 2 as i32;
        let mut work : Vec<F> = vec![C64::new(0.0,0.0);lwork as usize];
        let mut info=0 as i32;
        unsafe{
            let qrslice = qr.as_slice_memory_order_mut().expect("Memory not contiguous in input matrix");
            zgeqrf(m,n,qrslice,lda,&mut tau,&mut work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sgeqrf failed with info={}",info)[..]);
        QRFact { m : m as i32, n : n as i32, qr : qr, tau : tau, work : work, lwork : lwork}
    }
    /// Solve Rx=b in-place with the array b.
    pub fn solve_r(&mut self,b : &mut Array2<C64>) -> () {
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
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let lda=self.m;
        let ldb={
            let shape = b.shape();
            shape[0]
        } as i32;

        let bslice=b.as_slice_memory_order_mut().expect("Memory for B in RX=B not contiguous");
        let mut info = 0 as i32;
        unsafe {
            ztrtrs(b'U',b'N',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
        }
        check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
    }

    /// Solve R^T x=b in-place with the array b.
    pub fn solve_rt(&mut self,b : &mut Array2<C64>) -> () {
        check_if_fortran_style(b).expect("Matrix B in RX=B not column-major ordered");
        let n = self.n;
        let nrhs = {
            let shape=b.shape();
            shape[1]
        } as i32;
        let k = std::cmp::min(self.qr.shape()[0],self.qr.shape()[0]);
        {
            let shape = b.shape();
            if shape[0] as i32 != n {
                panic!(format!("Error: R in R^T X=B has shape ({},{}) but B has shape ({},{})",k,n,shape[0],shape[1]));
            }
        }
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let lda=self.m;
        let ldb={
            let shape=b.shape();
            shape[0]
        } as i32;
        let bslice=b.as_slice_memory_order_mut().expect("Memory for B in RX=B not contiguous");
        let mut info = 0;
        unsafe {
            ztrtrs(b'U',b'T',b'N',n,nrhs,qrslice,lda,bslice,ldb,&mut info);
        }
        check_lapack_info(info).expect(&format!("strtrs failed with info={}",info)[..]);
    }

    /// Multiply Q^T * C in-place in C.
    pub fn mul_left_qt(&mut self,c : &mut Array2<C64>) -> () {
        check_if_fortran_style(c).expect("Matrix C in Q^T * C not column-major ordered");
        {
            let m = self.m;
            let shape=c.shape();
            if shape[0] as i32 != m {
                panic!(format!("Error: Q in Q^T has shape ({},{}) but C has shape ({},{})",m,m,shape[0],shape[1]));
            }
        }
        let m = self.m as i32;
        let n = c.shape()[1] as i32;
        let k = self.tau.len() as i32;
        let lda = self.qr.shape()[0] as i32;
        let ldc = c.shape()[0] as i32;
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for R in RX=B not contiguous");
        let cslice=c.as_slice_memory_order_mut().expect("Memory for C in Q^T * C not contiguous");
        let lwork=self.lwork;
        let mut info = 0 as i32;
        unsafe{ 
            zunmqr(b'L',b'T',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
    }

    /// Multiply Q * C in-place in C.
    pub fn mul_left_q(&mut self,c : &mut Array2<C64>) -> () {
        check_if_fortran_style(c).expect("Matrix C in Q^T * C not column-major ordered");
        {
            let m = self.m;
            let shape=c.shape();
            if shape[0] as i32 != m {
                panic!(format!("Error: Q in Q * C has shape ({},{}) but C has shape ({},{})",m,m,shape[0],shape[1]));
            }
        }
        let m = self.m as i32;
        let n = c.shape()[1] as i32;
        let k = self.tau.len() as i32;
        let lda = self.qr.shape()[0] as i32;
        let ldc = c.shape()[0] as i32;
        let qrslice=self.qr.as_slice_memory_order_mut().expect("Memory for Q in Q * C not contiguous");
        let cslice=c.as_slice_memory_order_mut().expect("Memory for C in Q * C not contiguous");
        let lwork=self.lwork;
        let mut info = 0 as i32;
        unsafe{ 
            zunmqr(b'L',b'N',m,n,k,qrslice,lda,&mut self.tau,cslice,ldc,&mut self.work,lwork,&mut info);
        }
        check_lapack_info(info).expect(&format!("sormqr failed with info={}",info)[..]);
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








}



