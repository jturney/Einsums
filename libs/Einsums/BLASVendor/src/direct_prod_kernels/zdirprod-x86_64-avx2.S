
    /*
     * Calling convention:
     * Integer and pointer arguments: rdi, rsi, rdx, rcx, r8, r9
     * Floating point: xmm0-7
     * The rest: stack
     */
    .section .note.GNU-stack,"",@progbits
    .text
    .global zdirprod_kernel_avx2
    .arch .avx2
    .type zdirprod_kernel_avx2, @function
zdirprod_kernel_avx2:
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
    movupd  %xmm0, -64(%rbp)
    movupd  %xmm1, -96(%rbp)

    // And finally, save some ymm registers.
    vmovupd %ymm2, -128(%rbp)
    vmovupd %ymm3, -160(%rbp)
    vmovupd %ymm4, -192(%rbp)
    vmovupd %ymm5, -224(%rbp)
    vmovupd %ymm6, -256(%rbp)
    vmovupd %ymm7, -288(%rbp)

    // Calculate the number of vectorizable elements.
    movq    -8(%rbp), %rax
    shrq    $1, %rax
    movq    -8(%rbp), %rcx
    andq    $1, %rcx

    // Store the number of unvectorizable elements.
    pushq   %rcx

    // Now, get the components of the scale factors.
    vbroadcastsd    -64(%rbp), %ymm0
    vbroadcastsd    -96(%rbp), %ymm1

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
    vmovupd (%rsi), %ymm2

    // Multiply x by the scale factor.
    // Shuffle x.
    vpermilpd   $0x5, %ymm2, %ymm3

    // Scale x.
    vmulpd  %ymm1, %ymm3, %ymm4
    vfmaddsub231pd  %ymm0, %ymm2, %ymm4

    // Load z.
    vmovupd (%rbx), %ymm6

    // Load y.
    vmovupd (%rdi), %ymm2
    // Get the components.
    vpermilpd   $0xf, %ymm2, %ymm3
    vpermilpd   $0x0, %ymm2, %ymm2

    // Shuffle x.
    vpermilpd   $0x5, %ymm4, %ymm5

    // Multiply.
    vfmadd231pd %ymm3, %ymm5, %ymm6
    vfmaddsub231pd  %ymm2, %ymm4, %ymm6

    vmovupd %ymm6, (%rbx)

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
    movsd  (%rdi), %xmm2
    movsd  8(%rdi), %xmm3

    // Scale x.
    movsd  %xmm2, %xmm4

    mulsd   %xmm0, %xmm4
    vfnmadd231sd    %xmm1, %xmm3, %xmm4

    movsd  %xmm3, %xmm5

    mulsd   %xmm0, %xmm5
    vfmadd231sd %xmm1, %xmm2, %xmm5

    // Load z.
    movsd  (%rbx), %xmm6
    movsd  8(%rbx), %xmm7

    // Load y.
    movsd  (%rsi), %xmm2
    movsd  8(%rsi), %xmm3

    // Multiply and add.
    // Real-real
    vfmadd231sd %xmm2, %xmm4, %xmm6
    // Imag-imag
    vfnmadd231sd    %xmm3, %xmm5, %xmm6
    // Real-imag
    vfmadd231sd %xmm2, %xmm5, %xmm7
    // Imag-real
    vfmadd231sd %xmm3, %xmm4, %xmm7

    // Store.
    movsd  %xmm6, (%rbx)
    movsd  %xmm7, 8(%rbx)

    // Increment.
    incb    %cl

    leaq    16(%rdi), %rdi
    leaq    16(%rsi), %rsi
    leaq    16(%rbx), %rbx

    jmp .begin_unvector_loop_1
.end_unvector_loop_1:


    movq    -8(%rbp), %rdi
    movq    -16(%rbp), %rsi
    movq    -24(%rbp), %rdx
    movq    -32(%rbp), %rcx
    movq    -40(%rbp), %rbx
    movq    -48(%rbp), %rax
    movupd  -64(%rbp), %xmm0
    movupd  -96(%rbp), %xmm1
    vmovupd -128(%rbp), %ymm2
    vmovupd -160(%rbp), %ymm3
    vmovupd -192(%rbp), %ymm4
    vmovupd -224(%rbp), %ymm5
    vmovupd -256(%rbp), %ymm6
    vmovupd -288(%rbp), %ymm7

    addq    $288, %rsp
    popq    %rbp
    .cfi_def_cfa    rsp, 8
    retq

    .cfi_endproc
