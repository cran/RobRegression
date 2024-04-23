\name{Robust_regression}
\alias{Robust_regression}
\title{Robust_regression}

\description{This function gives robust estimates of the paramter of the Multivariate Linear regression with the help of the euclidean distance, or with the help of the Mahalanobis distance for some matrice Sigma. More precisely, the aim is to minimize

\eqn{
G(\hat{\beta}) = \mathbb{E}[ \| Y-X\hat{\beta} \|_{\Sigma}] + \lambda \| \hat{\beta}\|^{\text{ridge}}
}.

}
\usage{
Robust_regression(X,Y, Mat_Mahalanobis=diag(rep(1,ncol(Y))),
                  niter=50,lambda=0,c='default',method='Offline',
                  alpha=0.66,w=2,ridge=1,nlambda=50,
                  init=matrix(runif(ncol(X)*ncol(Y))-0.5,nrow=ncol(X),ncol=ncol(Y)),
                  epsilon=10^(-8), Mahalanobis_distance = FALSE,
                  par=TRUE,scale='none',tol=10^(-3))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
\item{X}{A (n,p)-matrix whose raws are the explaining data.}
\item{Y}{A (n,q)-matrix whose raws are the variables to be explained.}
\item{method}{The method used for estimating the parameter. Should be \code{method='Offline'} if the fix point algorithm is used, and \code{'Online'} if the (weighted) averaged stochastic gradient algorithm is used. Default is \code{'Offline'}.}
\item{Mat_Mahalanobis}{A (q,q)-matrix giving \eqn{\Sigma} for the Mahalanobis distance. Default is identity.}
\item{Mahalanobis_distance}{A logical telling if the Mahalanobis distance is used. Default is \code{FALSE}.}
\item{scale}{If a scaling is used.  \code{scale='robust'} should be used if a robust scaling of \code{Y} is desired.  Default is \code{'none'}}
\item{niter}{The maximum number of iteration if \code{method='Offline'}.}
\item{init}{A (p,q)-matrix which gives the initialization of the algorithm.}
\item{ridge}{The power of the penalty: i.e should be \code{2} if the squared norm is considered or \code{1} if the norm is considered.}
\item{lambda}{A vector giving the different studied penalizations. If \code{lambda='default'}, would be a vector of preselected penalizations.}
\item{nlambda}{The number of tested penalizations if \code{lambda='default'}.}
\item{par}{Is equal to \code{TRUE} if the parallelization of the algorithm for estimating robustly the variance of the noise is allowed.}
\item{c}{The constant in the stepsequence if the averaged stochastic gradient algorithm, i.e if \code{method='Online'}.}
\item{alpha}{A scalar between 1/2 and 1 used in the stepsequence for stochastic gradient algorithm if \code{method='Online'}.}
\item{w}{The power for the weighted averaged Robbins-Monro algorithm if \code{method='Online'}.}
\item{epsilon}{Stoping condition for the fix point algorithm if \code{method='Offline'}.}
\item{tol}{A scalar that avoid numerical problems if method='Offline'. Default is \code{10^(-3)}.}
}

\value{
A list with:
\item{beta}{A (p,q)-matrix giving the estimation of the parameters.}
\item{criterion}{A vector giving the loss for the different chosen \code{lambda}. If \code{sale='robust'}, it is calculated on the scaled data. }
\item{all_beta}{A list containing the different estimation of the parameters (with respect to the different coices of \code{lambda}).}
\item{lambda_opt}{A scalar giving the selected \code{lambda}.}
}
\references{
Godichon-Baggioni, A., Robin, S. and Sansonnet, L. (2023): A robust multivariate linear regression based on the Mahalanobis distance
}
\seealso{
See also \code{\link{Robust_Variance}}, \code{\link{Robust_Mahalanobis_regression}} and \code{\link{RobRegression-package}}.
}
\examples{
\donttest{
p=5
q=10
n=2000
mu=rep(0,q)
epsilon=mvtnorm::rmvnorm(n = n,mean = mu)
X=mvtnorm::rmvnorm(n=n,mean=rep(0,p))
beta=matrix(rnorm(p*q),ncol=q)
Y=X \%*\% beta+epsilon
Res_reg=Robust_regression(X,Y)
sum((Res_reg$beta-beta)^2)
}
}
