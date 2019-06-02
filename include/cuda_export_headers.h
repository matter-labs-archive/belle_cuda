#ifndef CUDA_EXPORT_HEADERS
#define CUDA_EXPORT_HEADERS

#define EXPORT __attribute__((visibility("default")))

struct EXPORT embedded_field;
struct EXPORT ec_point;
struct EXPORT affine_point;
struct EXPORT polynomial;

//-----------------------------------------------------------------------------------------------------------------------------------------------
//export basic parallel routines: finite field addition, substraction and multiplication; elliptic curve addiition and substraction
//-----------------------------------------------------------------------------------------------------------------------------------------------

void EXPORT field_add(const embedded_field*, const embedded_field*, embedded_field*, uint32_t);
void EXPORT field_sub(const embedded_field*, const embedded_field*, embedded_field*, uint32_t);
void EXPORT field_mul(const embedded_field*, const embedded_field*, embedded_field*, uint32_t);

void EXPORT ec_point_add(const ec_point*, const ec_point*, ec_point*, uint32_t);
void EXPORT ec_point_sub(const ec_point*, const ec_point*, ec_point*, uint32_t);

//-----------------------------------------------------------------------------------------------------------------------------------------------
//Multiexponentiation (based on Pippenger realization)
//-----------------------------------------------------------------------------------------------------------------------------------------------

ec_point EXPORT ec_multiexp(const affine_point*, const uint256_g*, uint32_t);

//-----------------------------------------------------------------------------------------------------------------------------------------------
//FFT routines
//-----------------------------------------------------------------------------------------------------------------------------------------------

void EXPORT FFT(const embedded_field*, embedded_field*, uint32_t);

void EXPORT iFFT(const embedded_field*, embedded_field*, uint32_t, const embedded_field&);

//------------------------------------------------------------------------------------------------------------------------------------------------
//polynomial arithmetic
//------------------------------------------------------------------------------------------------------------------------------------------------

// polynomial EXPORT poly_add(const& polynomial, const& polynomial);
// polynomial EXPORT poly_sub(const& polynomial, const& polynomial);
// polynomial EXPORT poly_mul(const& polynomial, const& polynomial);

#endif