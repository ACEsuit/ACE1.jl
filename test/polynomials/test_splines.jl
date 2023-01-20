

using ACE1, LinearAlgebra, Interpolations, BenchmarkTools, ForwardDiff


##

f = sin 
df = cos 

rr = range(0.0, stop=5.0, length=10_000)
ff = f.(rr)
spl = cubic_spline_interpolation(rr, ff)

# test 
tt = 5 * rand(100_000)
f_tt = f.(tt)
df_tt = df.(tt)
spl_tt = spl.(tt)
dspl_tt = Interpolations.gradient1.(Ref(spl), tt)


@show norm(f_tt - spl_tt, Inf);
@show norm(df_tt - dspl_tt, Inf);

## 

r = rand() * 5
@info("Performance of evaluation")
@btime $spl($r)

@info("Performance of gradient")
@btime Interpolations.gradient1($spl, $r)
@info("And with FOrwardDiff")
@btime ForwardDiff.derivative($spl, $r)

@btime $spl(ForwardDiff.Dual($r, 1))

df = spl(ForwardDiff.Dual(r, 1))