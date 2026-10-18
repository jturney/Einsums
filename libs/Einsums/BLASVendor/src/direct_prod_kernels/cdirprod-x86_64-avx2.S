
    /*
     * Calling convention:
     * Integer and pointer arguments: rdi, rsi, rdx, rcx, r8, r9
     * Floating point: xmm0-7
     * The rest: stack
     */
    .section .note.GNU-stack,"",@progbits
    .text
    .global cdirprod_kernel_avx2
    .arch .avx2
    .type cdirprod_kernel_avx2, @function
cdirprod_kernel_avx2:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa    rsp, 8
    movq    %rsp, %rbp
    .cfi_def_cfa    rbp, 8
    subq    $288, %rsp

    // Start by saving the integer arguments.
    movq    %rdi, -8(%rbp)
    movq    %rsi, -16(%rbp)
    movq    %rdx, -24(%rbp)
    movq    %rcx, -32(%rbp)

    // Save some more registers that we may use.
    movq    %rbx, -40(%rbp)
    movq    %rax, -48(%rbp)

    // Now for the floating point ones.
    movups  %xmm0, -64(%rbp)

    // And finally, save some ymm registers.
    vmovups %ymm1, -96(%rbp)
    vmovups %ymm2, -128(%rbp)
    vmovups %ymm3, -160(%rbp)
    vmovups %ymm4, -192(%rbp)
    vmovups %ymm5, -224(%rbp)
    vmovups %ymm6, -256(%rbp)
    vmovups %ymm7, -288(%rbp)

    // Calculate the number of vectorizable elements.
    movq    -8(%rbp), %rax
    shrq    $2, %rax
    movq    -8(%rbp), %rcx
    andq    $3, %rcx

    // Store the number of unvectorizable elements.
    pushq   %rcx

    // Now, get the components of the scale factors.
    vbroadcastss    -64(%rbp), %ymm0
    vbroadcastss    -60(%rbp), %ymm1

    // Load the addresses.
    movq    -16(%rbp), %rsi
    movq    -24(%rbp), %rdi
    movq    -32(%rbp), %rbx

    // Loop.
    xorq    %rcx, %rcx
.begin_vector_loop_1:
    cmpq    %rax, %rcx
    jge .end_vector_loop_1

    // Load x.
    vmovups (%rsi), %ymm2

    // Multiply x by the scale factor.
    // Shuffle x.
    vpermilps   $0xb1, %ymm2, %ymm3

    // Scale x.
    vmulps  %ymm1, %ymm3, %ymm4
    vfmaddsub231ps  %ymm0, %ymm2, %ymm4

    // Load z.
    vmovups (%rbx), %ymm6

    // Load y.
    vmovups (%rdi), %ymm2
    // Get the components.
    vpermilps   $0xf5, %ymm2, %ymm3
    vpermilps   $0xa0, %ymm2, %ymm2

    // Shuffle x.
    vpermilps   $0xb1, %ymm4, %ymm5

    // Multiply.
    vfmadd231ps %ymm3, %ymm5, %ymm6
    vfmaddsub231ps  %ymm2, %ymm4, %ymm6

    // Save.
    vmovups %ymm6, (%rbx)

    // Increment.
    incq    %rcx
    leaq    32(%rbx), %rbx
    leaq    32(%rsi), %rsi
    leaq    32(%rdi), %rdi

    jmp .begin_vector_loop_1
.end_vector_loop_1:

    // Load the number of unvectorizable elements.
    popq    %rax

    xorq    %rcx, %rcx
.begin_unvector_loop_1:
    cmpb    %al, %cl
    jge .end_unvector_loop_1

    // Load x.
    movss  (%rdi), %xmm2
    movss  4(%rdi), %xmm3

    // Scale x.
    movss  %xmm2, %xmm4

    mulss   %xmm0, %xmm4
    vfnmadd231ss    %xmm1, %xmm3, %xmm4

    movss  %xmm3, %xmm5

    mulss   %xmm0, %xmm5
    vfmadd231ss %xmm1, %xmm2, %xmm5

    // Load z.
    movss  (%rbx), %xmm6
    movss  4(%rbx), %xmm7

    // Load y.
    movss  (%rsi), %xmm2
    movss  4(%rsi), %xmm3

    // Multiply and add.
    // Real-real
    vfmadd231ss %xmm2, %xmm4, %xmm6
    // Imag-imag
    vfnmadd231ss    %xmm3, %xmm5, %xmm6
    // Real-imag
    vfmadd231ss %xmm2, %xmm5, %xmm7
    // Imag-real
    vfmadd231ss %xmm3, %xmm4, %xmm7

    // Store.
    movss  %xmm6, (%rbx)
    movss  %xmm7, 4(%rbx)

    // Increment.
    incb    %cl

    leaq    8(%rdi), %rdi
    leaq    8(%rsi), %rsi
    leaq    8(%rbx), %rbx

    jmp .begin_unvector_loop_1
.end_unvector_loop_1:


    movq    -8(%rbp), %rdi
    movq    -16(%rbp), %rsi
    movq    -24(%rbp), %rdx
    movq    -32(%rbp), %rcx
    movq    -40(%rbp), %rbx
    movq    -48(%rbp), %rax
    movups  -64(%rbp), %xmm0
    vmovups -96(%rbp), %ymm1
    vmovups -128(%rbp), %ymm2
    vmovups -160(%rbp), %ymm3
    vmovups -192(%rbp), %ymm4
    vmovups -224(%rbp), %ymm5
    vmovups -256(%rbp), %ymm6
    vmovups -288(%rbp), %ymm7

    addq    $288, %rsp
    popq    %rbp
    .cfi_def_cfa    rsp, 8
    retq

    .cfi_endproc
