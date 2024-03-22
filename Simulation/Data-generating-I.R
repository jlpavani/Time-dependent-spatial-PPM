library(mvtnorm); library(sf)

Output.Areas <- sf::read_sf(".", "RG2017_rgi"); remove(Output.Areas)
Output.MG <- Output.Areas[217:286,]

# **************************************************************************** #
# Data-generating settings from competing model
# **************************************************************************** #
narea = length(Output.MG$rgi); ntime = 100; ndisease = 2
N.all = narea * ntime * ndisease

# Regression structure
nbeta = 2; beta = c(0.1, 0.4, 0.2, 0.3)

X = array(NA, dim = c(narea, ntime, ndisease*nbeta))
X[,,1] = X[,,3] = sample(c(0,1), narea*ntime, replace = TRUE, prob = c(0.5, 0.5))
X[,,2] = X[,,4] = rmvnorm(ntime, mean = rep(0, narea), sigma = diag(0.5, narea))

x1 <- c(X[,1:ntime,1]); x2 <- c(X[,1:ntime,2])

Xbeta.mat <- cbind(beta[1] * x1 + beta[2] * x2, beta[3] * x1 + beta[4] * x2)
Xbeta.vec <- as.numeric(t(Xbeta.mat))

# Global spatial dependence
alpha = 0.8

# Adjacency matrix
W.nb <- spdep::poly2nb(Output.MG, row.names = 1:length(Output.MG)) 
adj_mat <- spdep::nb2mat(W.nb, style = "B")

# Spatial covariance matrix: Q(adj_mat, alpha)
Q.W <- alpha * (diag(apply(adj_mat, 2, sum)) - adj_mat) + (1 - alpha) * diag(rep(1,narea))
Q.W.inv <- solve(Q.W)

# Outcome covariance matrix: Sigma
Sigma <- 0.1 * array(c(5, 1, 1, 5), c(2,2))
Sigma.inv <- solve(Sigma)

# Spatial and between outcome covariance: D(rho) x Q(adj_mat, alpha) x Sigma
QSig.prec <- kronecker(Q.W, Sigma.inv)
QSig.var <- solve(QSig.prec)

# Global temporal dependence
qq = 2; rho1 = 0.6; rho2 = 0.3

# Spatio-temporal random effects: AR(2)
phi.t1 <- mvtnorm::rmvnorm(1, mean = rep(0, narea * ndisease), sigma = QSig.var)
phi.t2 <- mvtnorm::rmvnorm(1, mean = rho1 * phi.t1, sigma = QSig.var)
phi <- c(phi.t1, phi.t2)
for(i in 3:ntime){
  phi.t3 <- mvtnorm::rmvnorm(1, mean = rho1 * phi.t2 + rho2 * phi.t1, sigma = QSig.var)
  phi <- c(phi, phi.t3)
    
  phi.t1 <- phi.t2
  phi.t2 <- phi.t3
}
phi <- phi - mean(phi)
phi.mat <- matrix(phi, ncol = 2, byrow = TRUE)
  
# Response variable
Y <- rnorm(N.all, mean = Xbeta.vec + phi, sd = sqrt(10))
Y.mat <- matrix(Y, nrow = narea * ntime, ncol = ndisease, byrow = TRUE)
Y.cube <- array(NA, dim = c(narea, ntime, ndisease))
Y.cube[,,1] = matrix(Y.mat[,1], nrow = narea, ncol = ntime)
Y.cube[,,2] = matrix(Y.mat[,2], nrow = narea, ncol = ntime)
  
Umat1 = Umat2 = array(0, dim = c(narea, ntime, qq))
for(i in 1:narea){
  Umat1[i,2:ntime,1] = Y.cube[i, 1:(ntime-1), 1]
  Umat1[i,3:ntime,2] = Y.cube[i, 1:(ntime-2), 1]
    
  Umat2[i,2:ntime,1] = Y.cube[i, 1:(ntime-1), 2]
  Umat2[i,3:ntime,2] = Y.cube[i, 1:(ntime-2), 2]
}
Umat = array(0, dim = c(narea, ntime, qq*ndisease))
Umat[,,1:qq] = Umat1; Umat[,,(qq+1):(2*qq)] = Umat2