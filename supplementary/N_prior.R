library(NPrior)
run_exp <- function(n,p,d,seed){
  set.seed(seed)
  timestart<-Sys.time()
  X <- matrix(rnorm(n*p), nrow = n, ncol = p)
  corr  = 0
  if (corr ==1){
    ex = matrix(0,p,p)
    for (i in 1:p){
      for (j in 1:p){
        ex[i,j] = .5^abs(i-j)
      }
    }
    X = X%*%chol(ex) 
  }
  #X <- diag(n)
  
  w0 <- c(rep(5,d), rep(0,p-d))
  sigma2 <- 1
  y <- X%*%w0+rnorm(n)*sigma2
  fit1 <- NPrior_run(X, y,N=20000,prior = "SpSL-L", eta = 50,method="exact")
  nonzero <- apply(fit1$ThetaSamples, 2, function(x) sum(abs(x)>.1))
  #plot(nonzero[3000:5000],type='l')
  #write.csv(nonzero,'nprior',row.names=FALSE)
  mse <- apply(fit1$ThetaSamples, 2, function(x) sum((x-w0)^2))
  idx = seq(1,length(nonzero),5)
  plot(nonzero[idx])
  library(coda)
  neff <- function(x){
    ac = acf(x,lag.max = length(x),plot = FALSE)
    sums=0.
    nn = length(x)
    for (k in 1:length(ac)){
      sums <- sums+ (nn-k) * ac$acf[k]/nn
    }
    return(1/(1+2*sums))
  } 
  ESS <- numeric(5)
  for (j in 1:5){
    t1 = fit1$ThetaSamples[j,10000:15000]
    idx = seq(1,length(t1),5)
    t1 = t1[idx] 
    ESS[j] <- neff(t1)
  }
  print(c('Rep:',str(seed),ESS))
  timestop<-Sys.time()
  return(c(ESS, mse, timestop - timestart))
}
rep_t <- 20
ESS <- matrix(0,nrow=rep_t, ncol=5)
mse <- numeric(rep_t)
time_r <- numeric(rep_t)
for (i in 1:rep_t){
  A <- run_exp(200,500,5,i)
  ESS[i,] <- A[1]
  mse[i] <- A[2]
  time_r[i] <- A[3]
}