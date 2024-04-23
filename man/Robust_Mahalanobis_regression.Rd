\name{Robust_Mahalanobis_regression}
\alias{Robust_Mahalanobis_regression}
\title{Robust_Mahalanobis_regression}

\description{
We propose here a function which enables to provide a robust estimation of the parameters of Multivariate Gaussian Linear Models of the form \eqn{Y = X \beta + \epsilon} where \eqn{\epsilon} is a 0-mean Gaussian vector of variance \eqn{\Sigma}. In addition, one can aslo consider a low-rank variance of the form \eqn{\Sigma = C + \sigma I} where \eqn{\sigma} is a positive scalar and \eqn{C} is a matrix of rank \eqn{d}. More precisely, the aim is to minimize the functional

\eqn{G_\lambda(\hat{\beta}) = \mathbb{E}\left(\| Y-X\hat{\beta} \|_{\Sigma^{-1}}\right) + \lambda \|\hat{\beta}\|^{\text{Ridge}}}.
}

\usage{
Robust_Mahalanobis_regression(X, Y, alphaRM=0.66, alphareg=0.66, w=2, lambda=0,
                              creg='default', K=2:30, par=TRUE, epsilon=10^(-8),
                              method_regression='Offline', niter_regression=50,
                              cRM='default', mc_sample_size='default',
                              method_MCM='Weiszfeld', methodMC='Robbins',
                              niterMC=50, ridge=1, eps_vp=10^(-4), nlambda=50,
                              scale='none', tol=10^(-3))
}
%- maybe also 'usage' for other objects documented here.

\arguments{
\item{X}{A \eqn{(n,p)}-matrix whose rows are the explaining data.}
\item{Y}{A \eqn{(n,q)}-matrix whose rows are the variables to be explained.}
\item{method_regression}{The method used for estimating the parameter. Should be \code{method_regression='Offline'} if the fix point algorithm is used, and \code{method_regression='Online'} if the (weighted) averaged stochastic gradient algorithm is used. Default is \code{'Offline'}.}
\item{niter_regression}{The maximum number of regression iterations if the fix point algorithm is used, i.e. if \code{method_regression='Offline'}.}
\item{epsilon}{Stoping condition for the fix point algorithm if \code{method_regression='Offline'}.}
\item{scale}{If a scaling is used. \code{scale='robust'} should be used if a robust scaling of \code{Y} is desired.  Default is \code{'none'}.}
\item{ridge}{The power of the penalty: i.e. should be \code{2} if the squared norm is considered or \code{1} if the norm is considered.}
\item{lambda}{A vector giving the different studied penalizations. If \code{lambda='default'}, would be a vector of preselected penalizations.}
\item{par}{Is equal to \code{T} if the parallelization of the algorithm for estimating robustly the variance of the noise is allowed.}
\item{nlambda}{The number of tested penalizations if \code{lambda='default'}.}
\item{alphaRM}{A scalar between 1/2 and 1 used in the stepsequence if the Robbins-Monro algorithm is used, i.e. if \code{methodMC='Robbins'}. Default is \code{0.66}.}
\item{alphareg}{A scalar between 1/2 and 1 used in the stepsequence for stochastic gradient algorithm if \code{method_regression='Online'}. Default is \code{0.66}.}
\item{w}{The power for the weighted averaged algorithms if \code{method_regression='Online'} or if \code{methodMC='Robbins'}.}
\item{creg}{The constant in the stepsequence if the averaged stochastic gradient algorithm is used, i.e. if \code{method='Online'}.}
\item{K}{A vector containing the possible values of \eqn{d}. The good \eqn{d} is chosen with the help of a penatly criterion if the length of \code{K} is larger than 10. Default is \code{ncol(X)}.}
\item{mc_sample_size}{The number of data generated for the Monte-Carlo method for estimating robustly the eigenvalues of the variance.}
\item{method_MCM}{The method chosen to estimate Median Covariation Matrix. Can be \code{'Weiszfeld'} if the Weiszfeld algorithm is used, or \code{'ASGD'} if one chooses the Averaged Stochastic Gradient Descent algorithm.}
\item{methodMC}{The method chosen to estimate robustly the variance. Can be \code{'Robbins'}, \code{'Grad'} or \code{'Fix'}.}
\item{niterMC}{The number of iterations for estimating robustly the variance of each class if \code{methodMC='Fix'} or \code{methodMC='Grad'}.}
\item{eps_vp}{The minimum values for the estimates of the eigenvalues of the Variance can take. Default is \code{10^-4}.}
\item{cRM}{The constant in the stepsequence if the Robbins-Monro algorithm is used to robustly estimate the variance, i.e. if \code{methodMC='Robbins'}.}
\item{tol}{A scalar that avoid numerical problems if method='Offline'. Default is \code{10^(-3)}.}
}

\value{
A list with:
\item{beta}{A \eqn{(p,q)}-matrix giving the estimation of the parameters of the MultivariateGaussian Linear Regression.}
\item{Residual_Variance}{A \eqn{(q,q)}-matrix giving the estimation of the variance of the residuals.}
\item{criterion}{A vector giving the loss for the different chosen \code{lambda}. If \code{scale='robust'}, it is calculated on the scaled data. }
\item{all_beta}{A list containing the different estimation of the parameters (with respect to the different choices of \code{lambda}).}
\item{lambda_opt}{A scalar giving the selected \code{lambda}.}
\item{variance_results}{A list giving the results on the variance of the noise obtained with the help of the function \code{Robust_Variance}. If \code{scale='robust'}, it is calculated on the scaled data. The details are given above.}
Details of the list \code{variance_results}:
\item{Sigma}{The robust estimation of the variance.}
\item{invSigma}{The robuste estimation of the inverse of the variance.}
\item{MCM}{The Median Covariation Matrix.}
\item{eigenvalues}{A vector containing the estimation of the \eqn{d+1} main eigenvalues of the variance, where \eqn{d+1} is the optimal choice belonging to \code{K}.}
\item{MCM_eigenvalues}{A vector containing the estimation of the \eqn{d+1} main eigenvalues of the Median Covariation Matrix, where \eqn{d+1} is the optimal choice belonging to \code{K}.}
\item{cap}{The result given for capushe for selecting \eqn{d} if the length of \code{K} is larger than 10.}
\item{reduction_results}{A list containing the results for all possible \code{K}.}
}

\references{
Cardot, H., Cenac, P. and Zitt, P-A. (2013). Efficient and fast estimation of the geometric median in Hilbert spaces with an averaged stochastic gradient algorithm. \emph{Bernoulli}, 19, 18-43.

Cardot, H. and Godichon-Baggioni, A. (2017). Fast Estimation of the Median Covariation Matrix with Application to Online Robust Principal Components Analysis. \emph{Test}, 26(3), 461-480

Vardi, Y. and Zhang, C.-H. (2000). The multivariate L1-median and associated data depth. \emph{Proc. Natl. Acad. Sci. USA}, 97(4):1423-1426.
}

\seealso{
See also \code{\link{Robust_Variance}}, \code{\link{Robust_regression}} and \code{\link{RobRegression-package}}.
}

\examples{
\donttest{
p=5
q=10
n=2000
mu=rep(0,q)
Sigma=diag(c(q,rep(0.1,q-1)))
epsilon=mvtnorm::rmvnorm(n = n,mean = mu,sigma = Sigma)
X=mvtnorm::rmvnorm(n=n,mean=rep(0,p))
beta=matrix(rnorm(p*q),ncol=q)
Y=X \%*\% beta+epsilon
Res_reg=Robust_Mahalanobis_regression(X,Y,par=FALSE)
sum((Res_reg$beta-beta)^2)
}
}
