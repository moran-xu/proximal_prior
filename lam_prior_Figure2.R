p <- 10000
beta <- rnorm(p)
lam_list <- seq(0,3,.01)
w <- numeric(length(lam_list))
for (i in 1:length(lam_list)){
  lam = lam_list[i]
  theta <- sign(beta)*(abs(beta)-lam)*((abs(beta)-lam)>0)
  w[i] <- sqrt(sum((theta-beta)^2))/sqrt(sum((beta)^2))
}
plot(w,lam_list,type='l')

ww <- runif(p)
f <- approxfun(lam_list,w)
Density <- numeric(length(lam_list)-1)
for (i in 1:length(lam_list)-1){
  Density[i] <- (f(lam_list[i+1])-f(lam_list[i]))/(lam_list[i+1]-lam_list[i])
}
lam <- lam_list[1:length(lam_list)-1]
plot(lam,Density,type='l')
