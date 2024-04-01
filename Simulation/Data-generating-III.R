library(mvtnorm); library(sf); library(sn)

Output.Areas <- sf::read_sf(".", "RG2017_rgi"); remove(Output.Areas)
Output.MG <- Output.Areas[217:286,]

# **************************************************************************** #
# Data-generating settings from proposed model
# **************************************************************************** #
narea = length(Output.MG$rgi); ntime = 100; ndisease = 2

# Regression structure
nbeta = 2; beta = c(0.1, 0.4, 0.2, 0.3)

X = array(NA, dim = c(narea, ntime, ndisease*nbeta))
X[,,1] = X[,,3] = sample(c(0,1), narea*ntime, replace = TRUE, prob = c(0.5, 0.5))
X[,,2] = X[,,4] = rmvnorm(ntime, mean = rep(0, narea), sigma = diag(0.5, narea))

# Adjacency matrix
W.nb <- spdep::poly2nb(Output.MG, row.names = 1:length(Output.MG)) 
adj_mat <- spdep::nb2mat(W.nb, style = "B")

# Global spatial dependence
alpha = c(0.8, 0.8)

# Spatial covariance matrix: Q(adj_mat, alpha)
Q1 <- solve(diag(apply(adj_mat, 2, sum)) - alpha[1] * adj_mat)
Q2 <- solve(diag(apply(adj_mat, 2, sum)) - alpha[2] * adj_mat)

# Spatial random effects
phi1 <- mvtnorm::rmvnorm(1, mean = rep(0, narea), sigma = Q1)
phi2 <- mvtnorm::rmvnorm(1, mean = rep(0, narea), sigma = Q2)
phi = c(phi1, phi2); phi <- phi - mean(phi)

# Cluster structure - 1 cluster
partition = rep(1, narea)
ncl = length(unique(partition))

nrho = qq = 2
rho_star = matrix(c(1.6, -0.7, 1.6, -0.7), ndisease*nrho, ncl)
var_dta_star = 0.1

rho = matrix(NA, ndisease*nrho, narea)
for(j in 1:ncl){
  rho[,partition == j] = rho_star[,j]
}

# Cluster structure - 2 clusters
partition = c(2,2,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,1)
ncl = length(unique(partition))

nrho = qq = 2
rho_star = matrix(c(1.6, -0.7, 1.6, -0.7, 0.9, -0.1, 0.9, -0.1), ndisease*nrho, ncl)
var_dta_star = 0.1

rho = matrix(NA, ndisease*nrho, narea)
for(j in 1:ncl){
  rho[,partition == j] = rho_star[,j]
}

# Cluster structure - 3 clusters
partition = c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
ncl = length(unique(partition))

nrho = qq = 2
rho_star = matrix(c(1.6, -0.7, 1.6, -0.7, 0.3, 0.1, 0.3, 0.1,
                    0.9, -0.1, 0.9, -0.1), ndisease*nrho, ncl)
var_dta_star = 0.1

rho = matrix(NA, ndisease*nrho, narea)
for(j in 1:ncl){
  rho[,partition == j] = rho_star[,j]
}

var_dta = matrix(c(var_dta_star, 0.05, 0.05, var_dta_star), 2, 2)
  
# Response variable
dta = array(NA, dim = c(narea, ntime, ndisease))
for(i in 1:narea){
  
  # tt = 1
  m = c(phi[i] + X[i,1,1:2] %*% beta[1:2], phi[i+narea] + X[i,1,3:4] %*% beta[3:4])
  dta[i,1,] = sn::rmsn(1, xi = m, Omega = var_dta, alpha = c(1,10))
  
  # tt = 2
  m = c(phi[i] + X[i,2,1:2] %*% beta[1:2] + dta[i,1,1] * rho[1,i],
        phi[i+narea] + X[i,2,3:4] %*% beta[3:4] + dta[i,1,2] * rho[3,i])
  dta[i,2,] = sn::rmsn(1, xi = m, Omega = var_dta, alpha = c(1,10))
  
  # tt > qq
  for(tt in (qq+1):ntime){
    m = c(phi[i] + X[i,tt,1:2] %*% beta[1:2] + dta[i,tt-(1:2),1] %*% rho[1:2,i],
          phi[i+narea] + X[i,tt,3:4] %*% beta[3:4] + dta[i,tt-(1:2),2] %*% rho[3:4,i])
    dta[i,tt,] = sn::rmsn(1, xi = m, Omega = var_dta, alpha = c(1,10))
  }
}

Umat1 = Umat2 = array(0, dim = c(narea, ntime, qq))
for(i in 1:narea){
  Umat1[i,2:ntime,1] = dta[i, 1:(ntime-1), 1]
  Umat1[i,3:ntime,2] = dta[i, 1:(ntime-2), 1]
  
  Umat2[i,2:ntime,1] = dta[i, 1:(ntime-1), 2]
  Umat2[i,3:ntime,2] = dta[i, 1:(ntime-2), 2]
}
Umat = array(0, dim = c(narea, ntime, qq*ndisease))
Umat[,,1:nrho] = Umat1; Umat[,,(nrho+1):(2*nrho)] = Umat2