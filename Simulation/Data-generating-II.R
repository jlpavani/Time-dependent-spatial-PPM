library(mvtnorm); library(sf); library(blockmatrix); library(Matrix)

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

# Spatial parameters
alpha = c(0.5, 0.5); eta0_21 = 1; eta1_21 = 0.1

# Distance matrix
MG.coord = coordinates(Output.MG)
MG.latrange = round(quantile(MG.coord[,2], c(0.25, 0.75)))
MG.albersproj = mapproj::mapproject(MG.coord[,1], MG.coord[,2], projection = "albers", param = MG.latrange)
projmat = cbind(MG.albersproj$x, MG.albersproj$y)

perm = order(MG.albersproj$x + MG.albersproj$y)
adj_mat = adj_mat[perm, perm]
projmat = projmat[perm,]

dmat = as.matrix(dist(projmat))
dmat = dmat/mean(dmat[which(adj_mat==1)])

# Spatial covariance matrix
G1 = alpha[1]^dmat; G2 = alpha[2]^dmat
A = diag(eta0_21, narea) + eta1_21 * adj_mat
L = as.matrix(blockmatrix(names = c("I1","A","0","I2"), I1 = diag((ndisease-1)*narea),
                          A = A, I2 = diag(narea), dim=c(2,2)))
G = as.matrix(bdiag(G1, G2))
V = L %*% G %*% t(L)

# Spatio-temporal random effects
phi = c(rmvnorm(1, mean = rep(0, ndisease*narea), sigma = 0.0001*solve(V)))

# Cluster structure - 1 cluster
partition = rep(1, narea)
ncl = length(unique(partition))

nrho = qq = 2
rho_star = matrix(c(1.6, -0.7, 1.6, -0.7), ndisease*nrho, ncl)
var_dta_star = 0.001

rho = matrix(NA, ndisease*nrho, narea)
var_dta = rep(NA, narea)
for(j in 1:ncl){
  rho[,partition == j] = rho_star[,j]
  var_dta[partition == j] = var_dta_star[j]
}

# Cluster structure - 2 clusters
partition = c(2,2,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,1)
ncl = length(unique(partition))

nrho = qq = 2
rho_star = matrix(c(1.6, -0.7, 1.6, -0.7, 0.9, -0.1, 0.9, -0.1), ndisease*nrho, ncl)
var_dta_star = c(0.001, 0.001)

rho = matrix(NA, ndisease*nrho, narea)
var_dta = rep(NA, narea)
for(j in 1:ncl){
  rho[,partition == j] = rho_star[,j]
  var_dta[partition == j] = var_dta_star[j]
}

# Cluster structure - 3 clusters
partition = c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
ncl = length(unique(partition))

nrho = qq = 2
rho_star = matrix(c(1.6, -0.7, 1.6, -0.7, 0.3, 0.1, 0.3, 0.1,
                    0.9, -0.1, 0.9, -0.1), ndisease*nrho, ncl)
var_dta_star = c(0.001, 0.001, 0.001)

rho = matrix(NA, ndisease*nrho, narea)
var_dta = rep(NA, narea)
for(j in 1:ncl){
  rho[,partition == j] = rho_star[,j]
  var_dta[partition == j] = var_dta_star[j]
}

# Response variable
dta = array(NA, dim = c(narea, ntime, ndisease))
for(i in 1:narea){
  
  # tt = 1
  mean1 = phi[i] + X[i,1,1:2] %*% beta[1:2]
  dta[i,1,1] = rnorm(1, mean = mean1, sd = sqrt(var_dta[i]))
  
  mean2 = phi[i+narea] + X[i,1,3:4] %*% beta[3:4]
  dta[i,1,2] = rnorm(1, mean = mean2, sd = sqrt(var_dta[i]))
  
  # tt = 2
  mean1 = phi[i] + X[i,2,1:2] %*% beta[1:2] + dta[i,1,1] * rho[1,i]
  dta[i,2,1] = rnorm(1, mean = mean1, sd = sqrt(var_dta[i]))
  
  mean2 = phi[i+narea] + X[i,2,3:4] %*% beta[3:4] + dta[i,1,2] * rho[3,i]
  dta[i,2,2] = rnorm(1, mean = mean2, sd = sqrt(var_dta[i]))
  
  # tt > qq
  for(tt in (qq+1):(ntime+npred)){
    mean1 = phi[i] + X[i,tt,1:2] %*% beta[1:2] + dta[i,tt-(1:2),1] %*% rho[1:2,i]
    dta[i,tt,1] = rnorm(1, mean = mean1, sd = sqrt(var_dta[i]))
    
    mean2 = phi[i+narea] + X[i,tt,3:4] %*% beta[3:4] + dta[i,tt-(1:2),2] %*% rho[3:4,i]
    dta[i,tt,2] = rnorm(1, mean = mean2, sd = sqrt(var_dta[i]))
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