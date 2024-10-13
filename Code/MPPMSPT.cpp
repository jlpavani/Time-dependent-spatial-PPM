// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppDist)]] 

#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h> 
#include <Rcpp.h>
#include <vector>
#include <iostream>
#include <mvnorm.h>
#include <wishart.h>
#include <truncnorm.h>

using namespace Rcpp;
using namespace std;

// DAGAR components - number of neighbors
List adj_mat_index(arma::mat adj_mat){
	
	int narea = adj_mat.n_cols, acc = 0;
	arma::mat mat_low(narea, narea, arma::fill::zeros);
	mat_low.zeros();

	mat_low = trimatl(adj_mat);
	
	arma::colvec n_nei(narea, arma::fill::zeros);
	n_nei = sum(mat_low, 1);
	
	arma::colvec adj_index(narea, arma::fill::zeros);
	for(int i = 0; i < narea; i++){
 		acc += n_nei(i);
 		adj_index(i) = acc;
 	}
	
	int d = sum(n_nei), k = 0;
	arma::colvec nei(d, arma::fill::zeros);
	for(int i = 0; i < narea; i++){
		for(int j = 0; j < narea; j++){
			if(mat_low(i, j) == 1){
				nei(k) = j + 1;
				k += 1;
			}
		}
	}
	
	List L = List::create( _["n_nei"] = n_nei, _["neighbors"] = nei, _["adjacency_ends"] = adj_index );
	return L;
}

// Matrix of neighbors that don't belong to the same cluster: One represents that i and j are neighbors, but they don´t belong to the same cluster, i.e., it's an active area
arma::mat neighbor_mat_cpp(arma::mat adj_mat, arma::rowvec cl_id){
	
	int narea = adj_mat.n_rows;
	arma::mat neig_mat(narea, narea, arma::fill::zeros);
	
	for(int i = 0; i < narea; i++){
		for(int j = 0; j < narea; j++){
			if(adj_mat(i,j) == 1 && cl_id(i) == cl_id(j)){
				neig_mat(i,j) = 0;
			}else{
				neig_mat(i,j) = adj_mat(i,j);
			}	
		}
	}
	return neig_mat;
}

// Check if k belongs to the vector vec
bool belong(arma::uword k, arma::uvec vec){
  
    double su = 0.0;
	int si = vec.size();
    for(int i = 0; i < si; i++){
		if(k == vec(i)){
			su += 1;
		}
	}
    if(su != 0){
        return true;
    }else{
        return false;
    }
}

// Sum of the boundary lengths of the areas that S contains
arma::rowvec bound_cluster_cpp(arma::mat neig_mat, int ncl, arma::rowvec cl_id){
	
	int narea = neig_mat.n_rows;
	arma::rowvec b_area(narea, arma::fill::zeros), b_cluster(ncl, arma::fill::zeros);
	b_area = sum(neig_mat, 0);
	
	for(int j = 1; j < (ncl + 1); j++){
		for(int i = 0; i < narea; i++){
			if(cl_id(i) == j){
				b_cluster(j - 1) += b_area(i);
			}		
		}
	}
	return b_cluster;
}

// Logposterior distribution of alpha for the MH
double logpost_alpha1(arma::mat Q_var, arma::mat i_Q_var, double var_phi, arma::colvec phi, double alpha, double shape1_alpha, double shape2_alpha){
	
	return as_scalar(-0.5 * ( log(det(var_phi * Q_var)) + phi.t() * i_Q_var/var_phi * phi ) + (shape1_alpha - 1) * log(alpha) + (shape2_alpha - 1) * log(1 - alpha));
}

double logpost_alpha2(arma::mat Q_var, arma::mat i_Q_var, double var_phi, arma::colvec phi1, arma::colvec phi2, double alpha, double shape1_alpha, double shape2_alpha, arma::mat A){
	
	return as_scalar(-0.5 * ( log(det(var_phi * Q_var)) + (phi2 - A*phi1).t() * i_Q_var/var_phi * (phi2 - A*phi1) ) + (shape1_alpha - 1) * log(alpha) + (shape2_alpha - 1) * log(1 - alpha));
}

// [[Rcpp::export]]
List myPPM(arma::cube dta, arma::mat adj_mat,                  							// Data: response - matrix(narea, ntime), adjacency matrix - matrix(narea, narea)
		   arma::cube Ucube, arma::cube Xcube,                  						// covariates design matrix - cube(narea, ntime, p), temporal design matrix cube(narea, ntime, q) for disease 1
		   arma::colvec beta, arma::colvec mu0_beta, arma::mat Sigma0_beta,   			// Hyperparameters of beta
		   arma::colvec rho0, arma::colvec mu0_mu_rho, arma::mat Sigma0_mu_rho, 		// Hyperparameters of mu_rho
		   int df0_sigma_rho, arma::mat scale0_sigma_rho, 								// Hyperparameters of Sigma_rho
		   arma::rowvec omega, arma::colvec mu0_omega, arma::mat Sigma0_omega, 			// Hyperparameters of omega
		   arma::colvec phi, double shape_var_phi, double rate_var_phi,					// Hyperparameters of var_phi
		   arma::rowvec alpha, double shape1_alpha, double shape2_alpha, 				// Hyperparameters of alpha
		   arma::colvec var0, double xi0, double shape0_xi, double rate0_xi, double nu0,// Hyperparameters of xi and var_star
		   int niter, int nburn, int nthin,                   							// MCMC settings
		   int jumps, int m_neal, int q, double eta                     				// other objects
		){       	  
   
	RNGScope scope;
	
	// *************** 
	// Sample size
	// *************** 
	
	int ndisease = dta.n_slices, narea = adj_mat.n_rows, ntime = dta.n_cols;
	int	nbeta = Xcube.n_slices/ndisease, nrho = Ucube.n_slices/ndisease;
	
	// *************** 
	// Objects to save samples
	// *************** 

	arma::uvec seq_thin = arma::regspace<arma::uvec>(nburn, nthin, niter);
	arma::uvec seq_adap = arma::regspace<arma::uvec>(50, 50, nburn); 	  
	arma::uvec act_seq = arma::regspace<arma::uvec>(0, jumps, narea);   
	
	int nsample = ((niter - nburn)/nthin), jj = 0;
	arma::rowvec acceptance(ndisease, arma::fill::zeros);
	arma::colvec vec_ncl(nsample), vec_xi(nsample);
	arma::mat mat_alpha(nsample, ndisease), mat_omega(nsample, ndisease), mat_phi(nsample, narea*ndisease), mat_var_phi(nsample, ndisease), 
			  mat_cl_id(nsample, narea), mat_beta(nsample, nbeta*ndisease), mat_mu_rho(nsample, nrho*ndisease), mat_var_dta(nsample, narea*ndisease); 
	arma::cube cube_rho(nsample, nrho*ndisease, narea), cube_sigma_rho(nsample, nrho*ndisease, nrho*ndisease);      
	
	// *************** 
	// Initialize partition (with only one cluster)
	// *************** 
	
	int ncl = 1;
	arma::rowvec cl_id(narea, arma::fill::ones), cl_size(narea, arma::fill::zeros);
	cl_size(0) = narea;
	
	// *************** 
	// Initialize parameters according to their prior distribution
	// *************** 

	// *************** Regresion coefficients - beta
	arma::mat i_Sigma0_beta = arma::inv(Sigma0_beta);

	// *************** Spatial parameters and hyperparameters - phi, alpha, and omega
	arma::rowvec gammaa(ndisease), gammaa1(nburn), gammaa2(nburn); 
	gammaa(0) = log(alpha(0)/(1 - alpha(0))), gammaa(1) = log(alpha(1)/(1 - alpha(1)));
	gammaa1(0) = gammaa(0), gammaa2(0) = gammaa(1); 

	double gamma_inf = -4.9, gamma_sup = 4.9;	
	arma::colvec sd_gamma(ndisease, arma::fill::value(0.5)); 

	arma::mat i_Sigma0_omega = arma::inv(Sigma0_omega);
	
	List adj_index = adj_mat_index(adj_mat);
	arma::colvec adjacency_ends = adj_index["adjacency_ends"];
	arma::colvec n_nei = adj_index["n_nei"];
	arma::colvec neighbors = adj_index["neighbors"];

	arma::rowvec var_phi(ndisease, arma::fill::ones);
	arma::colvec mu_phi(narea, arma::fill::zeros), denom(narea, arma::fill::zeros), b(narea, arma::fill::zeros); 
	arma::mat I(narea, narea, arma::fill::eye), B(narea, narea, arma::fill::zeros), F(narea, narea, arma::fill::zeros);
	
	// i_Q_var is the precision matrix as defined by Datta and Gao
	arma::cube i_Q_var(narea, narea, ndisease, arma::fill::zeros), Q_var(narea, narea, ndisease, arma::fill::zeros);
	for(int d = 0; d < ndisease; d++){
	
		denom = 1 + (n_nei - 1) * pow(alpha(d), 2);
		b = alpha(d)/denom;
	
		F(0,0) = denom(0)/(1 - pow(alpha(d), 2));
		for(int i = 1; i < narea; i++){	
			for(int l = adjacency_ends(i - 1); l < adjacency_ends(i); l++){
				
				B(i, neighbors(l) - 1) = b(i);
			}
			
			F(i,i) = denom(i)/(1 - pow(alpha(d), 2));
		}

		i_Q_var(arma::span::all, arma::span::all, arma::span(d)) = (I - B).t() * F * (I - B);
		arma::mat i_Q_var_aux = i_Q_var(arma::span::all, arma::span::all, arma::span(d));
		
		Q_var(arma::span::all, arma::span::all, arma::span(d)) = arma::inv(i_Q_var_aux); 
		arma::mat Q_var_aux = Q_var(arma::span::all, arma::span::all, arma::span(d));
	}

	arma::mat i_Q_var1 = i_Q_var(arma::span::all, arma::span::all, arma::span(0)), i_Q_var2 = i_Q_var(arma::span::all, arma::span::all, arma::span(1));
	arma::mat Q_var1 = Q_var(arma::span::all, arma::span::all, arma::span(0)), Q_var2 = Q_var(arma::span::all, arma::span::all, arma::span(1));

	// *************** Cluster-specific parameters and hyperparameters *************** //
	// *************** Standard deviation hyperparameters - var_dta and xi
	arma::colvec var_star(narea*ndisease, arma::fill::ones), var_dta(narea*ndisease, arma::fill::ones);
	var_star(0) = var0(0);
	var_star(narea) = var0(1);

	for(int i = 0; i < narea; i++){
		var_dta(i) = var_star(0);
		var_dta(i+narea) = var_star(narea);
	}

	// *************** Mean hyperparameters - rho, mu_rho, and Sigma_rho
	arma::colvec mu_rho = rmvnorm(1, mu0_mu_rho, Sigma0_mu_rho).t();
	arma::colvec mu0_rho = mu_rho;
	
	arma::mat Sigma_rho = riwish(df0_sigma_rho, scale0_sigma_rho);
	arma::mat Sigma0_rho = Sigma_rho;
	arma::mat i_Sigma_rho = arma::inv(Sigma_rho), i_Sigma0_mu_rho = arma::inv(Sigma0_mu_rho);

	arma::mat rho_star(nrho*ndisease, narea, arma::fill::zeros), rho(nrho*ndisease, narea, arma::fill::zeros); 
	rho_star.col(0) = rho0;  
	rho = rho_star;

	// *************** Auxiliary structures
	arma::mat XX1(nbeta, nbeta, arma::fill::zeros), XX2(nbeta, nbeta, arma::fill::zeros), UU1(nrho, nrho, arma::fill::zeros), UU2(nrho, nrho, arma::fill::zeros),
			  X1_beta(narea, ntime, arma::fill::zeros), X2_beta(narea, ntime, arma::fill::zeros), U1_rho(narea, ntime, arma::fill::zeros), U2_rho(narea, ntime, arma::fill::zeros);
	arma::colvec X1(nbeta, arma::fill::zeros), X2(nbeta, arma::fill::zeros), U1(nrho, arma::fill::zeros), U2(nrho, arma::fill::zeros);

	for(int i = 0; i < narea; i++){
		for(int t = q; t < ntime; t++){

			X1 = Xcube(arma::span(i), arma::span(t), arma::span(0, (nbeta - 1)));				
			XX1 += (X1 * X1.as_row()/var_dta(i)); 
			X1_beta(i,t) = as_scalar(X1.as_row() * beta.rows(0, (nbeta - 1)));	

			X2 = Xcube(arma::span(i), arma::span(t), arma::span(nbeta, (nbeta*ndisease - 1)));				
			XX2 += (X2 * X2.as_row()/var_dta(i+narea)); 
			X2_beta(i,t) = as_scalar(X2.as_row() * beta.rows(nbeta, (nbeta*ndisease - 1)));
			
			U1 = Ucube(arma::span(i), arma::span(t), arma::span(0, (nrho - 1)));				
			UU1 += (U1 * U1.as_row());
			U1_rho(i,t) = as_scalar(U1.as_row() * rho_star(arma::span(0, (nrho - 1)), arma::span(ncl - 1)));	

			U2 = Ucube(arma::span(i), arma::span(t), arma::span(nrho, (nrho*ndisease - 1)));				
			UU2 += (U2 * U2.as_row());
			U2_rho(i,t) = as_scalar(U2.as_row() * rho_star(arma::span(nrho, (nrho*ndisease - 1)), arma::span(ncl - 1)));				
		}
	}				

	// ***************
	// Gibbs Sampling
	// ***************

	for(int iter = 1; iter < niter; iter++){
	
		// ******************************************************************
		// 1st step: Update cluster-specific hyperparameters
		// Update xi
		double sum_i_var = 0.0;
		for(int d = 0; d < ndisease; d++){
			for(int j = 1; j < (ncl + 1); j++){
				sum_i_var += 1/var_star((j - 1) + d*narea);
			}
		}
		double scale_xi_post = 1/(rate0_xi + nu0 * sum_i_var);
		double xi = Rf_rgamma(shape0_xi, scale_xi_post);
	
		// mu_rho
		arma::mat aux_mu_rho(nrho*ndisease, nrho*ndisease, arma::fill::zeros);
		aux_mu_rho = arma::inv(ncl * i_Sigma_rho + i_Sigma0_mu_rho);
		
 		arma::mat i_Sigma_mu_rho_post(nrho*ndisease, nrho*ndisease, arma::fill::zeros);
		i_Sigma_mu_rho_post = (aux_mu_rho + aux_mu_rho.t())/2;
	
		arma::colvec mu_mu_rho_post(nrho*ndisease, arma::fill::zeros);
		mu_mu_rho_post = i_Sigma_mu_rho_post * (i_Sigma_rho * sum(rho_star, 1) + i_Sigma0_mu_rho * mu0_mu_rho); // sum(rho_star, 1) means sum by rows	

		mu_rho.zeros();
		mu_rho = rmvnorm(1, mu_mu_rho_post, i_Sigma_mu_rho_post).t();	

		// Sigma_rho
		int df_sigma_rho_post = df0_sigma_rho + ncl;
		double sum_rhomu = 0.0;
		for(int j = 1; j < (ncl + 1); j++){
			sum_rhomu += as_scalar((rho_star.col(j - 1) - mu_rho).t() * (rho_star.col(j - 1) - mu_rho));
		}
		arma::mat scale_sigma_rho_post = scale0_sigma_rho + sum_rhomu;

		Sigma_rho.zeros(), i_Sigma_rho.zeros();
		Sigma_rho = riwish(df_sigma_rho_post, scale_sigma_rho_post);	
        i_Sigma_rho = arma::inv(Sigma_rho);

		// ******************************************************************
		// 2nd step: Update partition using Neal's 8 algorithm and Hegarty and Barry (2008) cohesion function
		int kminus, h;
		for(int i = 0; i < narea; i++){
			
			// Matrix of neighbors that don't belong to the same cluster. It will contain only active areas (see definition on Hegarty and Barry (2008)).
			arma::mat neig_mat = neighbor_mat_cpp(adj_mat, cl_id);

			// Check if the ith area is an active area (Hegarty and Barry)
			// Conditions to enter the loop: To be an active area or belong to a given sequence
			if(sum(neig_mat.col(i)) != 0 || belong(i, act_seq)){
			 
				// When the selected area belongs to a non singleton, do:
				if(cl_size(cl_id(i) - 1) > 1){
		
					kminus = ncl;
					h = kminus + m_neal;
					cl_size(cl_id(i) - 1) = cl_size(cl_id(i) - 1) - 1;
				
					rho_star.cols(kminus, h - 1) = rmvnorm(m_neal, mu0_rho, Sigma0_rho).t();
					rho_star.cols(h, narea - 1).zeros();
		
					for(int l = kminus; l < h; l++){
						var_star(l) = 1/Rf_rgamma(nu0, 1/(nu0*xi0));
						var_star(l + narea) = 1/Rf_rgamma(nu0, 1/(nu0*xi0));
					}
					var_star.rows(h, narea - 1).zeros();
					var_star.rows(h + narea, ndisease*narea - 1).zeros();
					
				// When the selected area belongs to a singleton, do: 
				}else{ 
		
					kminus = ncl - 1;
					h = kminus + m_neal;
	
					// When the singleton isn't the last cluster, do:
					if(cl_id(i) < ncl){
	
						int id_aux = cl_id(i);
						cl_id = cl_id.replace(ncl, id_aux);
						cl_id(i) = ncl;
					
						arma::colvec rho_star_aux(nrho*ndisease);
						rho_star_aux = rho_star.col(id_aux - 1);
						rho_star.col(id_aux - 1) = rho_star.col(ncl - 1);
						rho_star.col(ncl - 1) = rho_star_aux;
					
						double var_star_aux = var_star(id_aux - 1);
						var_star(id_aux - 1) = var_star(ncl - 1);
						var_star(ncl - 1) = var_star_aux;
						
						var_star_aux = var_star(narea + (id_aux - 1));
						var_star(narea + (id_aux - 1)) = var_star(narea + (id_aux - 1));
						var_star(narea + (id_aux - 1)) = var_star_aux;
          
						cl_size(id_aux - 1) = cl_size(ncl - 1);
						cl_size(ncl - 1) = 1;
					} 
				
					cl_size(ncl - 1) = cl_size(ncl - 1) - 1; 	
					ncl = ncl - 1;
	
					// This occurs only when m_neal > 1
					if(h >= kminus + 2){
					
						rho_star.cols(kminus + 1, h - 1) = rmvnorm((m_neal - 1), mu0_rho, Sigma0_rho).t();
						rho_star.cols(h, narea - 1).zeros();
						
						for(int l = kminus + 1; l < h; l++){
							var_star(l) = 1/Rf_rgamma(nu0, 1/(nu0*xi0));
							var_star(l + narea) = 1/Rf_rgamma(nu0, 1/(nu0*xi0));
						}
						var_star.rows(h, narea - 1).zeros();
						var_star.rows(h + narea, ndisease*narea - 1).zeros();						
					}	
				}

				// Resample membership for ith area: selection probability
				arma::rowvec prob_aux(h, arma::fill::zeros), like(h, arma::fill::zeros), cl_minus(narea, arma::fill::zeros), 
							 cl_plus(narea, arma::fill::zeros), bound_clus(ncl, arma::fill::zeros), bound_area(narea, arma::fill::zeros);
				double cohe_minus, cohe_plus;
		
				for(int j = 1; j < (h + 1); j++){
				    
					cl_minus.zeros(), cl_plus.zeros(), bound_clus.zeros(), bound_area.zeros();

					// Generate values from the likelihood distribution that will be used to build selection probabilities
					arma::mat V(ndisease, ndisease, arma::fill::eye);
					for(int t = q; t < ntime; t++){ 
			  
						U1 = Ucube(arma::span(i), arma::span(t), arma::span(0, (nrho - 1)));
						U2 = Ucube(arma::span(i), arma::span(t), arma::span(nrho, (nrho*ndisease - 1)));

						arma::rowvec M = {X1_beta(i,t) + as_scalar(U1.as_row() * rho_star(arma::span(0, (nrho - 1)), arma::span(j - 1))) + phi(i),
										  X2_beta(i,t) + as_scalar(U2.as_row() * rho_star(arma::span(nrho, (nrho*ndisease - 1)), arma::span(j - 1))) + phi(i + narea)};

						arma::rowvec D = {dta(i,t,0), dta(i,t,1)};

						V(0,0) = var_star(j - 1);
						V(1,1) = var_star(narea + j - 1);

						arma::vec dno = dmvnorm(D, M.as_col(), V, true);
						like(j - 1) += dno(0);
					}
					
					// Calculate selection probabilities based on cohesion function 
					if (j <= kminus){
           
						cl_minus = cl_id;
						cl_plus = cl_id;
	
						// When the ith area belongs to the cluster, do:
						if(cl_id(i) == j){
						
							// When the jth cluster is a singleton, do:
							if(cl_size(j - 1) == 1){
							
								cohe_minus = 0 * log(eta);
								bound_area = sum(neighbor_mat_cpp(adj_mat, cl_id), 0);
								cohe_plus = bound_area(i) * log(eta);
						
							// When the jth cluster after removing ith area is a singleton, do:
							}else if(cl_size(j - 1) == 2){
							
								cl_minus(i) = j - 1;
								int l = 0;
								while(cl_id(l) != j){
									l += 1;
								}

								bound_area = sum(neighbor_mat_cpp(adj_mat, cl_minus), 0);
								cohe_minus = bound_area(l) * log(eta);
								bound_clus = bound_cluster_cpp(adj_mat, ncl, cl_id);
								cohe_plus = bound_clus(j - 1) * log(eta);	 
						
							// When the jth cluster isn't a singleton anyway, do:
							}else{
							
								cl_minus(i) = j - 1;
								bound_clus = bound_cluster_cpp(adj_mat, ncl, cl_minus);
								cohe_minus = bound_clus(j - 1) * log(eta);
							
								bound_clus = bound_cluster_cpp(adj_mat, ncl, cl_id);
								cohe_plus = bound_clus(j - 1) * log(eta);	
							}
		
						// When the ith area doesn't belong to the jth cluster, so cl_minus = cl_id, do:
						}else{

							bound_clus = bound_cluster_cpp(adj_mat, ncl, cl_id);
							cohe_minus = bound_clus(j - 1) * log(eta);
							cl_plus(i) = j;
							bound_clus = bound_cluster_cpp(adj_mat, ncl, cl_plus);
							cohe_plus = bound_clus(j - 1) * log(eta);	
						} 
					
						prob_aux(j - 1) = like(j - 1) + cohe_plus - cohe_minus;

					}else{

						bound_area = sum(neighbor_mat_cpp(adj_mat, cl_id), 0);
						prob_aux(j - 1) = like(j - 1) + bound_area(i) * log(eta) - log(m_neal);
					} 
				} 
		
				arma::uvec components = arma::linspace<arma::uvec>(1, h, h);
				arma::rowvec probs(h);
		
				probs = exp(prob_aux - max(prob_aux));
				probs = probs/sum(probs);
		
				int choice = RcppArmadillo::sample(components, 1, true, probs.t()).at(0);
	
				// When choice is an existing value (parameter) the number of clusters remains the same, so do:
				if(choice <= ncl){
					cl_id(i) = choice;
					cl_size(cl_id(i) - 1) = cl_size(cl_id(i) - 1) + 1;
      
				// When choice is a new value (auxiliary component) the number of cluster increases by one unit, so do:
				}else{
					ncl = ncl + 1;
					cl_id(i) = ncl;
					cl_size(ncl - 1) = 1;
				} 					 	
			}
		}

		// ******************************************************************
		// 3rd step: Update cluster-specific parameters and hyperparameters
		// Update variance - var_star
		var_star.zeros();
		for(int j = 1; j < (ncl + 1); j++){
			
			double shape_dta_post = nu0 + (cl_size(j - 1)*(ntime - q))/2;
		
			double dtaXbetaUrhophi1 = 0.0, dtaXbetaUrhophi2 = 0.0;
			X1.zeros(), X1_beta.zeros(), X2.zeros(); X2_beta.zeros();
			for(int i = 0; i < narea; i++){
				if(cl_id(i) == j){
					for(int t = q; t < ntime; t++){		
						
						X1 = Xcube(arma::span(i), arma::span(t), arma::span(0, (nbeta - 1)));				
						X1_beta(i,t) = as_scalar(X1.as_row() * beta.rows(0, (nbeta - 1)));
						dtaXbetaUrhophi1 += pow((dta(i,t,0) - X1_beta(i,t) - U1_rho(i,t) - phi(i)), 2);						
			
						X2 = Xcube(arma::span(i), arma::span(t), arma::span(nbeta, (nbeta*ndisease - 1)));				
						X2_beta(i,t) = as_scalar(X2.as_row() * beta.rows(nbeta, (nbeta*ndisease - 1)));	
						dtaXbetaUrhophi2 += pow((dta(i,t,1) - X2_beta(i,t) - U2_rho(i,t) - phi(i + narea)), 2);
					}
				}
			}

			var_star(j - 1) = 1/Rf_rgamma(shape_dta_post, 1/(dtaXbetaUrhophi1/2 + nu0*xi));	
			var_star((j - 1) + narea) = 1/Rf_rgamma(shape_dta_post, 1/(dtaXbetaUrhophi2/2 + nu0*xi));	
		}

		// rho_star
		rho_star.zeros(), var_dta.zeros(), rho.zeros(); 
		for(int j = 1; j < (ncl + 1); j++){
	
			arma::colvec dtaXbetaphiU(nrho*ndisease, arma::fill::zeros);
			U1.zeros(), U2.zeros(), UU1.zeros(), UU2.zeros();
			for(int i = 0; i < narea; i++){
				if(cl_id(i) == j){
					for(int t = q; t < ntime; t++){
					
						U1 = Ucube(arma::span(i), arma::span(t), arma::span(0, (nrho - 1)));
						UU1 += (U1 * U1.as_row())/var_star(j - 1);						
	
						U2 = Ucube(arma::span(i), arma::span(t), arma::span(nrho, (nrho*ndisease - 1)));					
						UU2 += (U2 * U2.as_row())/var_star((j - 1) + narea);

						dtaXbetaphiU.rows(0, (nrho - 1)) += ((dta(i,t,0) - X1_beta(i,t) - phi(i)) * U1)/var_star(j - 1);
						dtaXbetaphiU.rows(nrho, (ndisease*nrho - 1)) += ((dta(i,t,1) - X2_beta(i,t) - phi(i + narea)) * U2)/var_star((j - 1) + narea);
					}
				}
			} 
			
			arma::mat UU(nrho*ndisease, nrho*ndisease, arma::fill::zeros);
			UU(arma::span(0, (nrho - 1)), arma::span(0, (nrho - 1))) = UU1;
			UU(arma::span(nrho, (nrho*ndisease - 1)), arma::span(nrho, (nrho*ndisease - 1))) = UU2;
			
			arma::mat aux_rho(nrho*ndisease, nrho*ndisease, arma::fill::zeros);
			aux_rho = arma::inv(UU + i_Sigma_rho);

			arma::mat i_Sigma_rho_post(nrho*ndisease, nrho*ndisease, arma::fill::zeros); 
			i_Sigma_rho_post = (aux_rho + aux_rho.t())/2;

			arma::colvec mu_rho_post(nrho*ndisease, arma::fill::zeros);
			mu_rho_post = i_Sigma_rho_post * (dtaXbetaphiU + (i_Sigma_rho * mu_rho));
		
			rho_star.col(j - 1) = rmvnorm(1, mu_rho_post, i_Sigma_rho_post).t();
			
			// Update U_rho for the following calculations 
			U1.zeros(), U2.zeros(), U1_rho.zeros(), U2_rho.zeros();
			for(int i = 0; i < narea; i++){
				if(cl_id(i) == j){
					for(int t = q; t < ntime; t++){	
					
						U1 = Ucube(arma::span(i), arma::span(t), arma::span(0, (nrho - 1)));			
						U1_rho(i,t) = as_scalar(U1.as_row() * rho_star(arma::span(0, (nrho - 1)), arma::span(ncl - 1)));	

						U2 = Ucube(arma::span(i), arma::span(t), arma::span(nrho, (nrho*ndisease - 1)));			
						U2_rho(i,t) = as_scalar(U2.as_row() * rho_star(arma::span(nrho, (nrho*ndisease - 1)), arma::span(ncl - 1)));
					}
					
					var_dta(i) = var_star(j - 1);
					var_dta(i + narea) = var_star((j - 1) + narea);

					rho.col(i) = rho_star.col(j - 1);
				}	
			}
		}

		// ******************************************************************
		// 4th step: Update spatial parameters following DAGAR structure

		// Update omega - 2 diseases
		arma::mat delta(narea, ndisease, arma::fill::zeros);
		delta.col(0) = phi.rows(0, narea - 1);
		delta.col(1) = trimatl(adj_mat) * phi.rows(0, narea - 1);
	
		arma::mat Sigma_omega_post = delta.t() * i_Q_var2/var_phi(1) * delta + i_Sigma0_omega;
		arma::mat aux_Sigma_omega = arma::inv(Sigma_omega_post);
		arma::mat i_Sigma_omega_post = (aux_Sigma_omega + aux_Sigma_omega.t())/2;
	
		arma::colvec mu_omega_post = i_Sigma_omega_post * (delta.t() * i_Q_var2/var_phi(1) * phi.rows(narea, (ndisease*narea - 1)) + i_Sigma0_omega * mu0_omega);
	
		omega.zeros();
		omega = rmvnorm(1, mu_omega_post, i_Sigma_omega_post);
		arma::mat A = omega(0) * I + omega(1) * adj_mat;  

		// Update alpha - 2 diseases
		if(belong(iter, seq_adap)){
			
			arma::vec log_sd = {log(stddev(gammaa1.cols(0, iter - 1))), log(stddev(gammaa2.cols(0, iter - 1)))};
			arma::vec eps = {0.01, 1/sqrt(iter)}; 
	
			if(acceptance(0)/iter < 0.44){
				sd_gamma(0) = exp(log_sd(0) + eps.max());
			}else{
				sd_gamma(0) = exp(log_sd(0) - eps.max());
			}
			
			if(acceptance(1)/iter < 0.44){
				sd_gamma(1) = exp(log_sd(1) + eps.max());
			}else{
				sd_gamma(1) = exp(log_sd(1) - eps.max());
			}
		}
		
		// 1st disease
		double gamma_candidate = r_truncnorm(gammaa(0), sd_gamma(0), gamma_inf, gamma_sup);
		double alpha_candidate = exp(gamma_candidate)/(1 + exp(gamma_candidate));

		denom.zeros(), b.zeros(), F.zeros(), B.zeros();
		denom = 1 + (n_nei - 1) * pow(alpha_candidate, 2);
		b = alpha_candidate/denom;
		
		F(0,0) = denom(0)/(1 - pow(alpha_candidate, 2));
		for(int i = 1; i < narea; i++){	
		
			for(int l = adjacency_ends(i - 1); l < adjacency_ends(i); l++){
				B(i, neighbors(l) - 1) = b(i);
			}
		
			F(i,i) = denom(i)/(1 - pow(alpha_candidate, 2));
		}
	
		arma::mat i_Q_var_candidate = (I - B).t() * F * (I - B); 
		arma::mat Q_var_candidate = arma::inv(i_Q_var_candidate);
	
		double accept_prob = logpost_alpha1(Q_var_candidate, i_Q_var_candidate, var_phi(0), phi.rows(0, narea - 1), alpha_candidate, shape1_alpha, shape2_alpha) 
						   - log(alpha(0)*(1 - alpha(0))) + d_truncnorm(gammaa(0), gamma_candidate, sd_gamma(0), gamma_inf, gamma_sup, 1)
						   - logpost_alpha1(Q_var1, i_Q_var1, var_phi(0), phi.rows(0, narea - 1), alpha(0), shape1_alpha, shape2_alpha) 
						   + log(alpha_candidate*(1 - alpha_candidate)) - d_truncnorm(gamma_candidate, gammaa(0), sd_gamma(0), gamma_inf, gamma_sup, 1);
	
		if((log(R::runif(0, 1)) <= std::min(0.0, accept_prob))){
			alpha(0) = alpha_candidate;
			gammaa(0) = gamma_candidate;
			
			i_Q_var1.zeros(), Q_var1.zeros();
			i_Q_var1 = i_Q_var_candidate;
			Q_var1 = Q_var_candidate;
			
			acceptance(0) += 1;
		}	

		// 2nd disease
		gamma_candidate = 0, alpha_candidate = 0, accept_prob = 0;
		
		gamma_candidate = r_truncnorm(gammaa(1), sd_gamma(1), gamma_inf, gamma_sup);
		alpha_candidate = exp(gamma_candidate)/(1 + exp(gamma_candidate));
		
		denom.zeros(), b.zeros(), F.zeros(), B.zeros();
		denom = 1 + (n_nei - 1) * pow(alpha_candidate, 2);
		b = alpha_candidate/denom;
		
		F(0,0) = denom(0)/(1 - pow(alpha_candidate, 2));
		for(int i = 1; i < narea; i++){	
		
			for(int l = adjacency_ends(i - 1); l < adjacency_ends(i); l++){
				B(i, neighbors(l) - 1) = b(i);
			}

			F(i,i) = denom(i)/(1 - pow(alpha_candidate, 2));
		}
	
		i_Q_var_candidate.zeros(), Q_var_candidate.zeros();
		i_Q_var_candidate = (I - B).t() * F * (I - B);
		Q_var_candidate = arma::inv(i_Q_var_candidate);
		
		accept_prob = logpost_alpha2(Q_var_candidate, i_Q_var_candidate, var_phi(1), phi.rows(0, narea - 1), phi.rows(narea, (ndisease*narea - 1)), alpha_candidate, shape1_alpha, shape2_alpha, A) 
					- log(alpha(1)*(1 - alpha(1))) + d_truncnorm(gammaa(1), gamma_candidate, sd_gamma(1), gamma_inf, gamma_sup, 1)
					- logpost_alpha2(Q_var2, i_Q_var2, var_phi(1), phi.rows(0, narea - 1), phi.rows(narea, (ndisease*narea - 1)), alpha(1), shape1_alpha, shape2_alpha, A) 
					+ log(alpha_candidate*(1 - alpha_candidate)) - d_truncnorm(gamma_candidate, gammaa(1), sd_gamma(1), gamma_inf, gamma_sup, 1);

		if((log(R::runif(0, 1)) <= std::min(0.0, accept_prob))){
			alpha(1) = alpha_candidate;
			gammaa(1) = gamma_candidate;
			
			i_Q_var2.zeros(), Q_var2.zeros();
			i_Q_var2 = i_Q_var_candidate;
			Q_var2 = Q_var_candidate;
			
			acceptance(1) += 1;
		}	

		if(iter < nburn){
			gammaa1(iter) = gammaa(0);  
			gammaa2(iter) = gammaa(1); 
		}

		// Update phi - disease 1
		arma::colvec sum_YXbetaUrho1(narea, arma::fill::zeros), sum_YXbetaUrho2(narea, arma::fill::zeros);
		for(int t = q; t < ntime; t++){
			
			arma::colvec dta_aux = dta(arma::span::all, arma::span(t), arma::span(0));
			sum_YXbetaUrho1 += (dta_aux - X1_beta.col(t) - U1_rho.col(t))/var_dta.rows(0, (narea - 1));
			
			dta_aux = dta(arma::span::all, arma::span(t), arma::span(1));
			sum_YXbetaUrho2 += (dta_aux - X2_beta.col(t) - U2_rho.col(t))/var_dta.rows(narea, (ndisease*narea - 1));
		}  
		
		arma::mat Sigma_phi_post(narea * ndisease, narea * ndisease, arma::fill::zeros);
		arma::mat Sigma_phi1 = arma::diagmat((ntime - q)/var_dta.rows(0, narea - 1)) + i_Q_var1/var_phi(0) + A.t() * i_Q_var2/var_phi(1) * A;
		
		arma::mat aux_Sigma_phi1 = arma::inv(Sigma_phi1);
		arma::mat i_Sigma_phi1 = (aux_Sigma_phi1 + aux_Sigma_phi1.t())/2;
		
		arma::colvec mu_phi_post(narea * ndisease, arma::fill::zeros);
		mu_phi_post.rows(0, narea - 1) = i_Sigma_phi1 * (sum_YXbetaUrho1 + A.t() * i_Q_var2/var_phi(1) * phi.rows(narea, (ndisease*narea - 1)));

		// Update phi - disease 2
		arma::mat Sigma_phi2 = arma::diagmat((ntime - q)/var_dta.rows(narea, (ndisease*narea - 1))) + i_Q_var2/var_phi(1);
		
		arma::mat aux_Sigma_phi2 = arma::inv(Sigma_phi2);
		arma::mat i_Sigma_phi2 = (aux_Sigma_phi2 + aux_Sigma_phi2.t())/2;
	
		mu_phi_post.rows(narea, ndisease*narea - 1) = i_Sigma_phi2 * (sum_YXbetaUrho2 + i_Q_var2/var_phi(1) * A * phi.rows(0, (narea - 1)));

        Sigma_phi_post(arma::span(0, narea - 1), arma::span(0, narea - 1)) = i_Sigma_phi1;
		Sigma_phi_post(arma::span(narea, ndisease*narea - 1), arma::span(narea, ndisease*narea - 1)) = i_Sigma_phi2;
	
		phi.zeros();
		phi = rmvnorm(1, mu_phi_post, Sigma_phi_post).t();
		
		// ******************************************************************
		// 5th step: Update regression coefficients
		arma::colvec dtaUrhophiX(nbeta*ndisease, arma::fill::zeros);	
		X1.zeros(), XX1.zeros(), X2.zeros(), XX2.zeros();
		for(int i = 0; i < narea; i++){
			for(int t = q; t < ntime; t++){
				
				X1 = Xcube(arma::span(i), arma::span(t), arma::span(0, (nbeta - 1)));				
				XX1 += (X1 * X1.as_row()/var_dta(i)); 
	
				X2 = Xcube(arma::span(i), arma::span(t), arma::span(nbeta, (nbeta*ndisease - 1)));				
				XX2 += (X2 * X2.as_row()/var_dta(i + narea)); 
	
				dtaUrhophiX.rows(0, (nbeta - 1)) += ((dta(i,t,0) - U1_rho(i,t) - phi(i)) * X1)/var_dta(i);	
				dtaUrhophiX.rows(nbeta, (ndisease*nbeta - 1)) += ((dta(i,t,1) - U2_rho(i,t) - phi(i + narea)) * X2)/var_dta(i + narea);	
			}
		}
	
		arma::mat XX(nbeta*ndisease, nbeta*ndisease, arma::fill::zeros);
		XX(arma::span(0, (nbeta - 1)), arma::span(0, (nbeta - 1))) = XX1;
		XX(arma::span(nbeta, (nbeta*ndisease - 1)), arma::span(nbeta, (nbeta*ndisease - 1))) = XX2;
		
		arma::mat aux_beta(nbeta*ndisease, nbeta*ndisease, arma::fill::zeros);
		aux_beta = arma::inv(XX + i_Sigma0_beta);
		
		arma::mat i_Sigma_beta_post(nbeta*ndisease, nbeta*ndisease, arma::fill::zeros);
		i_Sigma_beta_post = (aux_beta + aux_beta.t())/2;
		
		arma::colvec mu_beta_post(nbeta*ndisease, arma::fill::zeros);
		mu_beta_post = i_Sigma_beta_post * (dtaUrhophiX + (i_Sigma0_beta * mu0_beta));
		
		beta.zeros();
		beta = rmvnorm(1, mu_beta_post, i_Sigma_beta_post).t();	

		// ******************************************************************
		// Save sample
		if(iter >= nburn){ 
			if(belong(iter, seq_thin)){
				
				vec_ncl(jj) = ncl;
				mat_cl_id.row(jj) = cl_id;

				mat_beta(arma::span(jj), arma::span::all) = beta.t();
	
				cube_rho(arma::span(jj), arma::span::all, arma::span::all) = rho(arma::span::all, arma::span::all);
				mat_mu_rho(arma::span(jj), arma::span::all) = mu_rho.as_row();	
				cube_sigma_rho(arma::span(jj), arma::span::all, arma::span::all) = Sigma_rho(arma::span::all, arma::span::all);
				
				mat_phi(arma::span(jj), arma::span::all) = phi.t();
				mat_alpha.row(jj) = alpha;
				mat_omega.row(jj) = omega; 
				mat_var_phi.row(jj) = var_phi;
				
				vec_xi(jj) = xi;
				mat_var_dta(arma::span(jj), arma::span::all) = var_dta.t();

				jj += 1;
			}
		}	
	} 
	
	List L = List::create( _["ncl"] = vec_ncl, _["cl_id"] = mat_cl_id, 
						   _["beta"] = mat_beta, _["acceptance"] = acceptance/niter, _["sd_gamma"] = sd_gamma, 
						   _["phi"] = mat_phi, _["alpha"] = mat_alpha, _["omega"] = mat_omega, _["var_phi"] = mat_var_phi, 
	                       _["rho"] = cube_rho, _["mu_rho"] = mat_mu_rho, _["sigma_rho"] = cube_sigma_rho, 
						   _["var_dta"] = mat_var_dta, _["xi"] = vec_xi );
	
	return L;
}

// [[Rcpp::export]]
List myPPM_DP(arma::cube dta, arma::mat adj_mat,                  							// Data: response - matrix(narea, ntime), adjacency matrix - matrix(narea, narea)
		   arma::cube Ucube, arma::cube Xcube,                  						// covariates design matrix - cube(narea, ntime, p), temporal design matrix cube(narea, ntime, q) for disease 1
		   arma::colvec beta, arma::colvec mu0_beta, arma::mat Sigma0_beta,   			// Hyperparameters of beta
		   arma::colvec rho0, arma::colvec mu0_mu_rho, arma::mat Sigma0_mu_rho, 		// Hyperparameters of mu_rho
		   int df0_sigma_rho, arma::mat scale0_sigma_rho, 								// Hyperparameters of Sigma_rho
		   arma::rowvec omega, arma::colvec mu0_omega, arma::mat Sigma0_omega, 			// Hyperparameters of omega
		   arma::colvec phi, double shape_var_phi, double rate_var_phi,					// Hyperparameters of var_phi
		   arma::rowvec alpha, double shape1_alpha, double shape2_alpha, 				// Hyperparameters of alpha
		   arma::colvec var0, double xi0, double shape0_xi, double rate0_xi, double nu0,// Hyperparameters of xi and var_star
		   int niter, int nburn, int nthin,                   							// MCMC settings
		   int jumps, int m_neal, int q, double m_DP                      				// other objects
		){       	  
   
    RNGScope scope;
	
	// *************** 
	// Sample size
	// *************** 
	
	int ndisease = dta.n_slices, narea = adj_mat.n_rows, ntime = dta.n_cols;
	int	nbeta = Xcube.n_slices/ndisease, nrho = Ucube.n_slices/ndisease;
	
	// *************** 
	// Objects to save samples
	// *************** 

	arma::uvec seq_thin = arma::regspace<arma::uvec>(nburn, nthin, niter); 
	arma::uvec seq_adap = arma::regspace<arma::uvec>(50, 50, nburn); 	   
	arma::uvec act_seq = arma::regspace<arma::uvec>(0, jumps, narea);      
	
	int nsample = ((niter - nburn)/nthin), jj = 0;
	arma::rowvec acceptance(ndisease, arma::fill::zeros);
	arma::colvec vec_ncl(nsample), vec_xi(nsample);
	arma::mat mat_alpha(nsample, ndisease), mat_omega(nsample, ndisease), mat_phi(nsample, narea*ndisease), mat_var_phi(nsample, ndisease), 
			  mat_cl_id(nsample, narea), mat_beta(nsample, nbeta*ndisease), mat_mu_rho(nsample, nrho*ndisease), mat_var_dta(nsample, narea*ndisease); 
	arma::cube cube_rho(nsample, nrho*ndisease, narea), cube_sigma_rho(nsample, nrho*ndisease, nrho*ndisease);      
	
	// *************** 
	// Initialize partition (with only one cluster)
	// *************** 
	
	int ncl = 1;
	arma::rowvec cl_id(narea, arma::fill::ones), cl_size(narea, arma::fill::zeros);
	cl_size(0) = narea;
	
	// *************** 
	// Initialize parameters according to their prior distribution
	// *************** 

	// *************** Regresion coefficients - beta
	arma::mat i_Sigma0_beta = arma::inv(Sigma0_beta);

	// *************** Spatial parameters and hyperparameters - phi, alpha, and omega
	arma::rowvec gammaa(ndisease), gammaa1(nburn), gammaa2(nburn); 
	gammaa(0) = log(alpha(0)/(1 - alpha(0))), gammaa(1) = log(alpha(1)/(1 - alpha(1)));
	gammaa1(0) = gammaa(0), gammaa2(0) = gammaa(1); 

	double gamma_inf = -4.9, gamma_sup = 4.9;	
	arma::colvec sd_gamma(ndisease, arma::fill::value(0.5)); 

	arma::mat i_Sigma0_omega = arma::inv(Sigma0_omega);
	
	List adj_index = adj_mat_index(adj_mat);
	arma::colvec adjacency_ends = adj_index["adjacency_ends"];
	arma::colvec n_nei = adj_index["n_nei"];
	arma::colvec neighbors = adj_index["neighbors"];

	arma::rowvec var_phi(ndisease, arma::fill::ones);
	arma::colvec mu_phi(narea, arma::fill::zeros), denom(narea, arma::fill::zeros), b(narea, arma::fill::zeros); 
	arma::mat I(narea, narea, arma::fill::eye), B(narea, narea, arma::fill::zeros), F(narea, narea, arma::fill::zeros);
	
	// i_Q_var is the precision matrix as defined by Datta and Gao
	arma::cube i_Q_var(narea, narea, ndisease, arma::fill::zeros), Q_var(narea, narea, ndisease, arma::fill::zeros);
	for(int d = 0; d < ndisease; d++){
	
		denom = 1 + (n_nei - 1) * pow(alpha(d), 2);
		b = alpha(d)/denom;
	
		F(0,0) = denom(0)/(1 - pow(alpha(d), 2));
		for(int i = 1; i < narea; i++){	
			for(int l = adjacency_ends(i - 1); l < adjacency_ends(i); l++){
				
				B(i, neighbors(l) - 1) = b(i);
			}
			
			F(i,i) = denom(i)/(1 - pow(alpha(d), 2));
		}

		i_Q_var(arma::span::all, arma::span::all, arma::span(d)) = (I - B).t() * F * (I - B);
		arma::mat i_Q_var_aux = i_Q_var(arma::span::all, arma::span::all, arma::span(d));
		
		Q_var(arma::span::all, arma::span::all, arma::span(d)) = arma::inv(i_Q_var_aux); 
		arma::mat Q_var_aux = Q_var(arma::span::all, arma::span::all, arma::span(d));
	}

	arma::mat i_Q_var1 = i_Q_var(arma::span::all, arma::span::all, arma::span(0)), i_Q_var2 = i_Q_var(arma::span::all, arma::span::all, arma::span(1));
	arma::mat Q_var1 = Q_var(arma::span::all, arma::span::all, arma::span(0)), Q_var2 = Q_var(arma::span::all, arma::span::all, arma::span(1));

	// *************** Cluster-specific parameters and hyperparameters *************** //
	// *************** Standard deviation hyperparameters - var_dta and xi
	arma::colvec var_star(narea*ndisease, arma::fill::ones), var_dta(narea*ndisease, arma::fill::ones);
	var_star(0) = var0(0);
	var_star(narea) = var0(1);

	for(int i = 0; i < narea; i++){
		var_dta(i) = var_star(0);
		var_dta(i+narea) = var_star(narea);
	}

	// *************** Mean hyperparameters - rho, mu_rho, and Sigma_rho
	arma::colvec mu_rho = rmvnorm(1, mu0_mu_rho, Sigma0_mu_rho).t();
	arma::colvec mu0_rho = mu_rho;
	
	arma::mat Sigma_rho = riwish(df0_sigma_rho, scale0_sigma_rho);
	arma::mat Sigma0_rho = Sigma_rho;
	arma::mat i_Sigma_rho = arma::inv(Sigma_rho), i_Sigma0_mu_rho = arma::inv(Sigma0_mu_rho);

	arma::mat rho_star(nrho*ndisease, narea, arma::fill::zeros), rho(nrho*ndisease, narea, arma::fill::zeros); 
	rho_star.col(0) = rho0;  
	rho = rho_star;

	// *************** Auxiliary structures
	arma::mat XX1(nbeta, nbeta, arma::fill::zeros), XX2(nbeta, nbeta, arma::fill::zeros), UU1(nrho, nrho, arma::fill::zeros), UU2(nrho, nrho, arma::fill::zeros),
			  X1_beta(narea, ntime, arma::fill::zeros), X2_beta(narea, ntime, arma::fill::zeros), U1_rho(narea, ntime, arma::fill::zeros), U2_rho(narea, ntime, arma::fill::zeros);
	arma::colvec X1(nbeta, arma::fill::zeros), X2(nbeta, arma::fill::zeros), U1(nrho, arma::fill::zeros), U2(nrho, arma::fill::zeros);

	for(int i = 0; i < narea; i++){
		for(int t = q; t < ntime; t++){

			X1 = Xcube(arma::span(i), arma::span(t), arma::span(0, (nbeta - 1)));				
			XX1 += (X1 * X1.as_row()/var_dta(i)); 
			X1_beta(i,t) = as_scalar(X1.as_row() * beta.rows(0, (nbeta - 1)));	

			X2 = Xcube(arma::span(i), arma::span(t), arma::span(nbeta, (nbeta*ndisease - 1)));				
			XX2 += (X2 * X2.as_row()/var_dta(i+narea)); 
			X2_beta(i,t) = as_scalar(X2.as_row() * beta.rows(nbeta, (nbeta*ndisease - 1)));
			
			U1 = Ucube(arma::span(i), arma::span(t), arma::span(0, (nrho - 1)));				
			UU1 += (U1 * U1.as_row());
			U1_rho(i,t) = as_scalar(U1.as_row() * rho_star(arma::span(0, (nrho - 1)), arma::span(ncl - 1)));	

			U2 = Ucube(arma::span(i), arma::span(t), arma::span(nrho, (nrho*ndisease - 1)));				
			UU2 += (U2 * U2.as_row());
			U2_rho(i,t) = as_scalar(U2.as_row() * rho_star(arma::span(nrho, (nrho*ndisease - 1)), arma::span(ncl - 1)));				
		}
	}				

	// ***************
	// Gibbs Sampling
	// ***************

	for(int iter = 1; iter < niter; iter++){
	
		// ******************************************************************
		// 1st step: Update cluster-specific hyperparameters
		// Update xi
		double sum_i_var = 0.0;
		for(int d = 0; d < ndisease; d++){
			for(int j = 1; j < (ncl + 1); j++){
				sum_i_var += 1/var_star((j - 1) + d*narea);
			}
		}
		double scale_xi_post = 1/(rate0_xi + nu0 * sum_i_var);
		double xi = Rf_rgamma(shape0_xi, scale_xi_post);
	
		// mu_rho
		arma::mat aux_mu_rho(nrho*ndisease, nrho*ndisease, arma::fill::zeros);
		aux_mu_rho = arma::inv(ncl * i_Sigma_rho + i_Sigma0_mu_rho);
		
 		arma::mat i_Sigma_mu_rho_post(nrho*ndisease, nrho*ndisease, arma::fill::zeros);
		i_Sigma_mu_rho_post = (aux_mu_rho + aux_mu_rho.t())/2;
	
		arma::colvec mu_mu_rho_post(nrho*ndisease, arma::fill::zeros);
		mu_mu_rho_post = i_Sigma_mu_rho_post * (i_Sigma_rho * sum(rho_star, 1) + i_Sigma0_mu_rho * mu0_mu_rho); // sum(rho_star, 1) means sum by rows	

		mu_rho.zeros();
		mu_rho = rmvnorm(1, mu_mu_rho_post, i_Sigma_mu_rho_post).t();	

		// Sigma_rho
		int df_sigma_rho_post = df0_sigma_rho + ncl;
		double sum_rhomu = 0.0;
		for(int j = 1; j < (ncl + 1); j++){
			sum_rhomu += as_scalar((rho_star.col(j - 1) - mu_rho).t() * (rho_star.col(j - 1) - mu_rho));
		}
		arma::mat scale_sigma_rho_post = scale0_sigma_rho + sum_rhomu;

		Sigma_rho.zeros(), i_Sigma_rho.zeros();
		Sigma_rho = riwish(df_sigma_rho_post, scale_sigma_rho_post);	
        i_Sigma_rho = arma::inv(Sigma_rho);

		// ******************************************************************
		// 2nd step: Update partition using Neal's 8 algorithm and Hegarty and Barry (2008) cohesion function
		int kminus, h;
		for(int i = 0; i < narea; i++){
			
			// Matrix of neighbors that don't belong to the same cluster. It will contain only active areas (see definition on Hegarty and Barry (2008)).
			arma::mat neig_mat = neighbor_mat_cpp(adj_mat, cl_id);

			// Check if the ith area is an active area (Hegarty and Barry)
			// Conditions to enter the loop: To be an active area - sum(neig_mat.col(i)) != 0 - or belong to a given sequence - belong(i, act_seq)
			if(sum(neig_mat.col(i)) != 0 || belong(i, act_seq)){
			 
				// When the selected area belongs to a non singleton, do:
				if(cl_size(cl_id(i) - 1) > 1){
		
					kminus = ncl;
					h = kminus + m_neal;
					cl_size(cl_id(i) - 1) = cl_size(cl_id(i) - 1) - 1;
				
					// Generate m_neal extra parameters from the prior defined by the model
					rho_star.cols(kminus, h - 1) = rmvnorm(m_neal, mu0_rho, Sigma0_rho).t();
					rho_star.cols(h, narea - 1).zeros();
		
					for(int l = kminus; l < h; l++){
						var_star(l) = 1/Rf_rgamma(nu0, 1/(nu0*xi0));
						var_star(l + narea) = 1/Rf_rgamma(nu0, 1/(nu0*xi0));
					}
					var_star.rows(h, narea - 1).zeros();
					var_star.rows(h + narea, ndisease*narea - 1).zeros();
					
				// When the selected area belongs to a singleton, do: 
				}else{ 
		
					kminus = ncl - 1; 
					h = kminus + m_neal;
	
					// When the singleton isn't the last cluster, do:
					if(cl_id(i) < ncl){
	
						int id_aux = cl_id(i);
						cl_id = cl_id.replace(ncl, id_aux);
						cl_id(i) = ncl;
					
						arma::colvec rho_star_aux(nrho*ndisease);
						rho_star_aux = rho_star.col(id_aux - 1);
						rho_star.col(id_aux - 1) = rho_star.col(ncl - 1);
						rho_star.col(ncl - 1) = rho_star_aux;
					
						double var_star_aux = var_star(id_aux - 1);
						var_star(id_aux - 1) = var_star(ncl - 1);
						var_star(ncl - 1) = var_star_aux;
						
						var_star_aux = var_star(narea + (id_aux - 1));
						var_star(narea + (id_aux - 1)) = var_star(narea + (id_aux - 1));
						var_star(narea + (id_aux - 1)) = var_star_aux;
          
						cl_size(id_aux - 1) = cl_size(ncl - 1);
						cl_size(ncl - 1) = 1;
					} 
				
					// When the singleton is the last cluster, no switching are needed, so just do:
					cl_size(ncl - 1) = cl_size(ncl - 1) - 1; 	
					ncl = ncl - 1;
	
					// This occurs only when m_neal > 1
					if(h >= kminus + 2){
					
						rho_star.cols(kminus + 1, h - 1) = rmvnorm((m_neal - 1), mu0_rho, Sigma0_rho).t();
						rho_star.cols(h, narea - 1).zeros();
						
						for(int l = kminus + 1; l < h; l++){
							var_star(l) = 1/Rf_rgamma(nu0, 1/(nu0*xi0));
							var_star(l + narea) = 1/Rf_rgamma(nu0, 1/(nu0*xi0));
						}
						var_star.rows(h, narea - 1).zeros();
						var_star.rows(h + narea, ndisease*narea - 1).zeros();						
					}	
				}

				// Resample membership for ith area: selection probability
				arma::rowvec prob_aux(h, arma::fill::zeros), like(h, arma::fill::zeros);
				for(int j = 1; j < (h + 1); j++){

					// Generate values from the likelihood distribution that will be used to build selection probabilities
					arma::mat V(ndisease, ndisease, arma::fill::eye);
					for(int t = q; t < ntime; t++){ 
			  
						U1 = Ucube(arma::span(i), arma::span(t), arma::span(0, (nrho - 1)));
						U2 = Ucube(arma::span(i), arma::span(t), arma::span(nrho, (nrho*ndisease - 1)));

						arma::rowvec M = {X1_beta(i,t) + as_scalar(U1.as_row() * rho_star(arma::span(0, (nrho - 1)), arma::span(j - 1))) + phi(i),
										  X2_beta(i,t) + as_scalar(U2.as_row() * rho_star(arma::span(nrho, (nrho*ndisease - 1)), arma::span(j - 1))) + phi(i + narea)};

						arma::rowvec D = {dta(i,t,0), dta(i,t,1)};

						V(0,0) = var_star(j - 1);
						V(1,1) = var_star(narea + j - 1);

						arma::vec dno = dmvnorm(D, M.as_col(), V, true);
						like(j - 1) += dno(0);
					}
					
					// Calculate selection probabilities based on cohesion function 
					if (j <= kminus){
           
						prob_aux(j - 1) = like(j - 1) + lgamma(cl_size(j - 1) + 1) - lgamma(cl_size(j - 1));

					}else{

						prob_aux(j - 1) = like(j - 1) + log(m_DP) + lgamma(cl_size(j - 1) + 1) - log(m_neal);
					} 
				} 
		
				arma::uvec components = arma::linspace<arma::uvec>(1, h, h);
				arma::rowvec probs(h);
		
				probs = exp(prob_aux - max(prob_aux));
				probs = probs/sum(probs);
		
				int choice = RcppArmadillo::sample(components, 1, true, probs.t()).at(0);
	
				// When choice is an existing value (parameter) the number of clusters remains the same, so do:
				if(choice <= ncl){
					cl_id(i) = choice;
					cl_size(cl_id(i) - 1) = cl_size(cl_id(i) - 1) + 1;
      
				// When choice is a new value (auxiliary component) the number of cluster increases by one unit, so do:
				}else{
					ncl = ncl + 1;
					cl_id(i) = ncl;
					cl_size(ncl - 1) = 1;
				} 					 	
			}
		}

		// ******************************************************************
		// 3rd step: Update cluster-specific parameters and hyperparameters
		// Update variance - var_star
		var_star.zeros();
		for(int j = 1; j < (ncl + 1); j++){
			
			double shape_dta_post = nu0 + (cl_size(j - 1)*(ntime - q))/2;
		
			double dtaXbetaUrhophi1 = 0.0, dtaXbetaUrhophi2 = 0.0;
			X1.zeros(), X1_beta.zeros(), X2.zeros(); X2_beta.zeros();
			for(int i = 0; i < narea; i++){
				if(cl_id(i) == j){
					for(int t = q; t < ntime; t++){		
						
						X1 = Xcube(arma::span(i), arma::span(t), arma::span(0, (nbeta - 1)));				
						X1_beta(i,t) = as_scalar(X1.as_row() * beta.rows(0, (nbeta - 1)));
						dtaXbetaUrhophi1 += pow((dta(i,t,0) - X1_beta(i,t) - U1_rho(i,t) - phi(i)), 2);						
			
						X2 = Xcube(arma::span(i), arma::span(t), arma::span(nbeta, (nbeta*ndisease - 1)));				
						X2_beta(i,t) = as_scalar(X2.as_row() * beta.rows(nbeta, (nbeta*ndisease - 1)));	
						dtaXbetaUrhophi2 += pow((dta(i,t,1) - X2_beta(i,t) - U2_rho(i,t) - phi(i + narea)), 2);
					}
				}
			}

			var_star(j - 1) = 1/Rf_rgamma(shape_dta_post, 1/(dtaXbetaUrhophi1/2 + nu0*xi));	
			var_star((j - 1) + narea) = 1/Rf_rgamma(shape_dta_post, 1/(dtaXbetaUrhophi2/2 + nu0*xi));	
		}

		// rho_star
		rho_star.zeros(), var_dta.zeros(), rho.zeros(); 
		for(int j = 1; j < (ncl + 1); j++){
	
			arma::colvec dtaXbetaphiU(nrho*ndisease, arma::fill::zeros);
			U1.zeros(), U2.zeros(), UU1.zeros(), UU2.zeros();
			for(int i = 0; i < narea; i++){
				if(cl_id(i) == j){
					for(int t = q; t < ntime; t++){
					
						U1 = Ucube(arma::span(i), arma::span(t), arma::span(0, (nrho - 1)));
						UU1 += (U1 * U1.as_row())/var_star(j - 1);						
	
						U2 = Ucube(arma::span(i), arma::span(t), arma::span(nrho, (nrho*ndisease - 1)));					
						UU2 += (U2 * U2.as_row())/var_star((j - 1) + narea);

						dtaXbetaphiU.rows(0, (nrho - 1)) += ((dta(i,t,0) - X1_beta(i,t) - phi(i)) * U1)/var_star(j - 1);
						dtaXbetaphiU.rows(nrho, (ndisease*nrho - 1)) += ((dta(i,t,1) - X2_beta(i,t) - phi(i + narea)) * U2)/var_star((j - 1) + narea);
					}
				}
			} 
			
			arma::mat UU(nrho*ndisease, nrho*ndisease, arma::fill::zeros);
			UU(arma::span(0, (nrho - 1)), arma::span(0, (nrho - 1))) = UU1;
			UU(arma::span(nrho, (nrho*ndisease - 1)), arma::span(nrho, (nrho*ndisease - 1))) = UU2;
			
			arma::mat aux_rho(nrho*ndisease, nrho*ndisease, arma::fill::zeros);
			aux_rho = arma::inv(UU + i_Sigma_rho);

			arma::mat i_Sigma_rho_post(nrho*ndisease, nrho*ndisease, arma::fill::zeros); 
			i_Sigma_rho_post = (aux_rho + aux_rho.t())/2;

			arma::colvec mu_rho_post(nrho*ndisease, arma::fill::zeros);
			mu_rho_post = i_Sigma_rho_post * (dtaXbetaphiU + (i_Sigma_rho * mu_rho));
		
			rho_star.col(j - 1) = rmvnorm(1, mu_rho_post, i_Sigma_rho_post).t();
			
			U1.zeros(), U2.zeros(), U1_rho.zeros(), U2_rho.zeros();
			for(int i = 0; i < narea; i++){
				if(cl_id(i) == j){
					for(int t = q; t < ntime; t++){	
					
						U1 = Ucube(arma::span(i), arma::span(t), arma::span(0, (nrho - 1)));			
						U1_rho(i,t) = as_scalar(U1.as_row() * rho_star(arma::span(0, (nrho - 1)), arma::span(ncl - 1)));	

						U2 = Ucube(arma::span(i), arma::span(t), arma::span(nrho, (nrho*ndisease - 1)));			
						U2_rho(i,t) = as_scalar(U2.as_row() * rho_star(arma::span(nrho, (nrho*ndisease - 1)), arma::span(ncl - 1)));
					}
					
					var_dta(i) = var_star(j - 1);
					var_dta(i + narea) = var_star((j - 1) + narea);

					rho.col(i) = rho_star.col(j - 1);
				}	
			}
		}

		// ******************************************************************
		// 4th step: Update spatial parameters following DAGAR structure

		// Update omega - 2 diseases
		arma::mat delta(narea, ndisease, arma::fill::zeros);
		delta.col(0) = phi.rows(0, narea - 1);
		delta.col(1) = trimatl(adj_mat) * phi.rows(0, narea - 1);
	
		arma::mat Sigma_omega_post = delta.t() * i_Q_var2/var_phi(1) * delta + i_Sigma0_omega;
		arma::mat aux_Sigma_omega = arma::inv(Sigma_omega_post);
		arma::mat i_Sigma_omega_post = (aux_Sigma_omega + aux_Sigma_omega.t())/2;
	
		arma::colvec mu_omega_post = i_Sigma_omega_post * (delta.t() * i_Q_var2/var_phi(1) * phi.rows(narea, (ndisease*narea - 1)) + i_Sigma0_omega * mu0_omega);
	
		omega.zeros();
		omega = rmvnorm(1, mu_omega_post, i_Sigma_omega_post);
		arma::mat A = omega(0) * I + omega(1) * adj_mat;  

		// Update alpha - 2 diseases
		if(belong(iter, seq_adap)){
			
			arma::vec log_sd = {log(stddev(gammaa1.cols(0, iter - 1))), log(stddev(gammaa2.cols(0, iter - 1)))};
			arma::vec eps = {0.01, 1/sqrt(iter)}; 
	
			if(acceptance(0)/iter < 0.44){
				sd_gamma(0) = exp(log_sd(0) + eps.max());
			}else{
				sd_gamma(0) = exp(log_sd(0) - eps.max());
			}
			
			if(acceptance(1)/iter < 0.44){
				sd_gamma(1) = exp(log_sd(1) + eps.max());
			}else{
				sd_gamma(1) = exp(log_sd(1) - eps.max());
			}
		}
		
		// 1st disease
		double gamma_candidate = r_truncnorm(gammaa(0), sd_gamma(0), gamma_inf, gamma_sup);
		double alpha_candidate = exp(gamma_candidate)/(1 + exp(gamma_candidate));

		denom.zeros(), b.zeros(), F.zeros(), B.zeros();
		denom = 1 + (n_nei - 1) * pow(alpha_candidate, 2);
		b = alpha_candidate/denom;
		
		F(0,0) = denom(0)/(1 - pow(alpha_candidate, 2));
		for(int i = 1; i < narea; i++){	
		
			for(int l = adjacency_ends(i - 1); l < adjacency_ends(i); l++){
				B(i, neighbors(l) - 1) = b(i);
			}
		
			F(i,i) = denom(i)/(1 - pow(alpha_candidate, 2));
		}
	
		arma::mat i_Q_var_candidate = (I - B).t() * F * (I - B); 
		arma::mat Q_var_candidate = arma::inv(i_Q_var_candidate);
	
		double accept_prob = logpost_alpha1(Q_var_candidate, i_Q_var_candidate, var_phi(0), phi.rows(0, narea - 1), alpha_candidate, shape1_alpha, shape2_alpha) 
						   - log(alpha(0)*(1 - alpha(0))) + d_truncnorm(gammaa(0), gamma_candidate, sd_gamma(0), gamma_inf, gamma_sup, 1)
						   - logpost_alpha1(Q_var1, i_Q_var1, var_phi(0), phi.rows(0, narea - 1), alpha(0), shape1_alpha, shape2_alpha) 
						   + log(alpha_candidate*(1 - alpha_candidate)) - d_truncnorm(gamma_candidate, gammaa(0), sd_gamma(0), gamma_inf, gamma_sup, 1);
	
		if((log(R::runif(0, 1)) <= std::min(0.0, accept_prob))){
			alpha(0) = alpha_candidate;
			gammaa(0) = gamma_candidate;
			
			i_Q_var1.zeros(), Q_var1.zeros();
			i_Q_var1 = i_Q_var_candidate;
			Q_var1 = Q_var_candidate;
			
			acceptance(0) += 1;
		}	

		// 2nd disease
		gamma_candidate = 0, alpha_candidate = 0, accept_prob = 0;
		
		gamma_candidate = r_truncnorm(gammaa(1), sd_gamma(1), gamma_inf, gamma_sup);
		alpha_candidate = exp(gamma_candidate)/(1 + exp(gamma_candidate));
		
		denom.zeros(), b.zeros(), F.zeros(), B.zeros();
		denom = 1 + (n_nei - 1) * pow(alpha_candidate, 2);
		b = alpha_candidate/denom;
		
		F(0,0) = denom(0)/(1 - pow(alpha_candidate, 2));
		for(int i = 1; i < narea; i++){	
		
			for(int l = adjacency_ends(i - 1); l < adjacency_ends(i); l++){
				B(i, neighbors(l) - 1) = b(i);
			}
		
			F(i,i) = denom(i)/(1 - pow(alpha_candidate, 2));
		}
	
		i_Q_var_candidate.zeros(), Q_var_candidate.zeros(); 
		i_Q_var_candidate = (I - B).t() * F * (I - B); 
		Q_var_candidate = arma::inv(i_Q_var_candidate); 
		
		accept_prob = logpost_alpha2(Q_var_candidate, i_Q_var_candidate, var_phi(1), phi.rows(0, narea - 1), phi.rows(narea, (ndisease*narea - 1)), alpha_candidate, shape1_alpha, shape2_alpha, A) 
					- log(alpha(1)*(1 - alpha(1))) + d_truncnorm(gammaa(1), gamma_candidate, sd_gamma(1), gamma_inf, gamma_sup, 1)
					- logpost_alpha2(Q_var2, i_Q_var2, var_phi(1), phi.rows(0, narea - 1), phi.rows(narea, (ndisease*narea - 1)), alpha(1), shape1_alpha, shape2_alpha, A) 
					+ log(alpha_candidate*(1 - alpha_candidate)) - d_truncnorm(gamma_candidate, gammaa(1), sd_gamma(1), gamma_inf, gamma_sup, 1);

		if((log(R::runif(0, 1)) <= std::min(0.0, accept_prob))){
			alpha(1) = alpha_candidate;
			gammaa(1) = gamma_candidate;
			
			i_Q_var2.zeros(), Q_var2.zeros();
			i_Q_var2 = i_Q_var_candidate;
			Q_var2 = Q_var_candidate;
			
			acceptance(1) += 1;
		}	

		if(iter < nburn){
			gammaa1(iter) = gammaa(0);  
			gammaa2(iter) = gammaa(1); 
		}

		// Update phi - disease 1
		arma::colvec sum_YXbetaUrho1(narea, arma::fill::zeros), sum_YXbetaUrho2(narea, arma::fill::zeros);
		for(int t = q; t < ntime; t++){
			
			arma::colvec dta_aux = dta(arma::span::all, arma::span(t), arma::span(0));
			sum_YXbetaUrho1 += (dta_aux - X1_beta.col(t) - U1_rho.col(t))/var_dta.rows(0, (narea - 1));
			
			dta_aux = dta(arma::span::all, arma::span(t), arma::span(1));
			sum_YXbetaUrho2 += (dta_aux - X2_beta.col(t) - U2_rho.col(t))/var_dta.rows(narea, (ndisease*narea - 1));
		}  
		
		arma::mat Sigma_phi_post(narea * ndisease, narea * ndisease, arma::fill::zeros);
		arma::mat Sigma_phi1 = arma::diagmat((ntime - q)/var_dta.rows(0, narea - 1)) + i_Q_var1/var_phi(0) + A.t() * i_Q_var2/var_phi(1) * A;
		
		arma::mat aux_Sigma_phi1 = arma::inv(Sigma_phi1);
		arma::mat i_Sigma_phi1 = (aux_Sigma_phi1 + aux_Sigma_phi1.t())/2;
		
		arma::colvec mu_phi_post(narea * ndisease, arma::fill::zeros);
		mu_phi_post.rows(0, narea - 1) = i_Sigma_phi1 * (sum_YXbetaUrho1 + A.t() * i_Q_var2/var_phi(1) * phi.rows(narea, (ndisease*narea - 1)));

		// Update phi - disease 2
		arma::mat Sigma_phi2 = arma::diagmat((ntime - q)/var_dta.rows(narea, (ndisease*narea - 1))) + i_Q_var2/var_phi(1);
		
		arma::mat aux_Sigma_phi2 = arma::inv(Sigma_phi2);
		arma::mat i_Sigma_phi2 = (aux_Sigma_phi2 + aux_Sigma_phi2.t())/2;
	
		mu_phi_post.rows(narea, ndisease*narea - 1) = i_Sigma_phi2 * (sum_YXbetaUrho2 + i_Q_var2/var_phi(1) * A * phi.rows(0, (narea - 1)));

        Sigma_phi_post(arma::span(0, narea - 1), arma::span(0, narea - 1)) = i_Sigma_phi1;
		Sigma_phi_post(arma::span(narea, ndisease*narea - 1), arma::span(narea, ndisease*narea - 1)) = i_Sigma_phi2;
	
		phi.zeros();
		phi = rmvnorm(1, mu_phi_post, Sigma_phi_post).t();
		
		// ******************************************************************
		// 5th step: Update regression coefficients
		arma::colvec dtaUrhophiX(nbeta*ndisease, arma::fill::zeros);	
		X1.zeros(), XX1.zeros(), X2.zeros(), XX2.zeros();
		for(int i = 0; i < narea; i++){
			for(int t = q; t < ntime; t++){
				
				X1 = Xcube(arma::span(i), arma::span(t), arma::span(0, (nbeta - 1)));				
				XX1 += (X1 * X1.as_row()/var_dta(i)); 
	
				X2 = Xcube(arma::span(i), arma::span(t), arma::span(nbeta, (nbeta*ndisease - 1)));				
				XX2 += (X2 * X2.as_row()/var_dta(i + narea)); 
	
				dtaUrhophiX.rows(0, (nbeta - 1)) += ((dta(i,t,0) - U1_rho(i,t) - phi(i)) * X1)/var_dta(i);	
				dtaUrhophiX.rows(nbeta, (ndisease*nbeta - 1)) += ((dta(i,t,1) - U2_rho(i,t) - phi(i + narea)) * X2)/var_dta(i + narea);	
			}
		}
	
		arma::mat XX(nbeta*ndisease, nbeta*ndisease, arma::fill::zeros);
		XX(arma::span(0, (nbeta - 1)), arma::span(0, (nbeta - 1))) = XX1;
		XX(arma::span(nbeta, (nbeta*ndisease - 1)), arma::span(nbeta, (nbeta*ndisease - 1))) = XX2;
		
		arma::mat aux_beta(nbeta*ndisease, nbeta*ndisease, arma::fill::zeros);
		aux_beta = arma::inv(XX + i_Sigma0_beta);
		
		arma::mat i_Sigma_beta_post(nbeta*ndisease, nbeta*ndisease, arma::fill::zeros);
		i_Sigma_beta_post = (aux_beta + aux_beta.t())/2;
		
		arma::colvec mu_beta_post(nbeta*ndisease, arma::fill::zeros);
		mu_beta_post = i_Sigma_beta_post * (dtaUrhophiX + (i_Sigma0_beta * mu0_beta));
		
		beta.zeros();
		beta = rmvnorm(1, mu_beta_post, i_Sigma_beta_post).t();	
	
		// ******************************************************************
		// Save sample
		if(iter >= nburn){ 
			if(belong(iter, seq_thin)){
				
				vec_ncl(jj) = ncl;
				mat_cl_id.row(jj) = cl_id;

				mat_beta(arma::span(jj), arma::span::all) = beta.t();
	
				cube_rho(arma::span(jj), arma::span::all, arma::span::all) = rho(arma::span::all, arma::span::all);
				mat_mu_rho(arma::span(jj), arma::span::all) = mu_rho.as_row();	
				cube_sigma_rho(arma::span(jj), arma::span::all, arma::span::all) = Sigma_rho(arma::span::all, arma::span::all);
				
				mat_phi(arma::span(jj), arma::span::all) = phi.t();
				mat_alpha.row(jj) = alpha;
				mat_omega.row(jj) = omega; 
				mat_var_phi.row(jj) = var_phi;
				
				vec_xi(jj) = xi;
				mat_var_dta(arma::span(jj), arma::span::all) = var_dta.t();

				jj += 1;
			}
		}	
	} 
	
	List L = List::create( _["ncl"] = vec_ncl, _["cl_id"] = mat_cl_id, 
						   _["beta"] = mat_beta, _["acceptance"] = acceptance/niter, _["sd_gamma"] = sd_gamma, 
						   _["phi"] = mat_phi, _["alpha"] = mat_alpha, _["omega"] = mat_omega, _["var_phi"] = mat_var_phi, 
	                       _["rho"] = cube_rho, _["mu_rho"] = mat_mu_rho, _["sigma_rho"] = cube_sigma_rho, 
						   _["var_dta"] = mat_var_dta, _["xi"] = vec_xi );
	
	return L;
}