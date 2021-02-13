# dissectionqr
Nested Dissection based sparse QR Factorization using Rust




# TODO

1. blas version 0.20.0 uses `num_traits`<0.3 whereas everything else uses `num_traits_>=0.3
    * when blas 0.21.0 becomes available switch to this version
    * use github link in cargo.toml for now
2. Implement iterators for tree types and use instead of accessing fields directly
    * Make fields private after this
3. Figure out how to make tests type-generic over `float32,float64,complex32,complex64`?
4. Figure out how to make `QRFact` type-generic over `float32,float64,complex32,complex64`?
5. Switch from using basic functions to construct some types to implementing `new()` method instead
