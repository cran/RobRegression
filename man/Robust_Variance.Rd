\name{Robust_Variance}
\alias{Robust_Variance}
\title{Robust_Variance}

\description{The aim is to provide a robust estimation of the variance for Guassian models with reduction dimension. More precisely we considering a q dimensional random vector  whose variance can be written as \eqn{\Sigma = C + \sigma I} where C is a matrix of rank d, with d possibly much smaller than q, \eqn{sigma} is a positive scalar, and I is the identity matrix.
}
\usage{
Robust_Variance(X,K=ncol(X),par=TRUE,alphaRM=0.75,
                c='default',w=2,mc_sample_size='default',
                methodMC='Robbins',niterMC=50,method_MCM='Weiszfeld',
                eps_vp=10^(-6))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
\item{X}{A matrix whose raws are the vector we want to estimate the variance.}
\item{K}{A vector containing the possible values of d. The 'good' d is chosen with the help of a penatly criterion if the length of K is larger than 10. Default is \code{ncol(X)}.}
\item{par}{Is equal to \code{TRUE} if the parallelization of the algorithm is allowed.}
\item{mc_sample_size}{The number of data generated for the Monte-Carlo method for estimating robustly the eigenvalues of the variance.}
\item{methodMC}{The method chosen to estimate robustly the variance. Can be \code{'Robbins'}, \code{'Grad'} or \code{'Fix'}. Default is \code{'Robbins'}.}
\item{niterMC}{The number of iterations for estimating robustly the variance of each class if \code{methodMC='Fix'} or \code{methodMC='Grad'}.}
\item{method_MCM}{The method chosen to estimate Median Covariation Matrix. Can be \code{'Weiszfeld'} or \code{'ASGD'}.}
\item{alphaRM}{A scalar between 1/2 and 1 used in the stepsequence for the Robbins-Monro method if \code{methodMC='Robbins'}.}
\item{c}{The constant in the stepsequence if \code{methodMC='Robbins'}.}
\item{w}{The power for the weighted averaged Robbins-Monro algorithm if \code{methodMC='Robbins'}. Default is \code{2}.}
\item{eps_vp}{The minimum values for the estimates of the eigenvalues of the Variance can take. Default is \code{10^-6}.}
}




\value{
A list with:
\item{Sigma}{The robust estimation of the variance.}
\item{invSigma}{The robuste estimation of the inverse of the variance.}
\item{MCM}{The Median Covariation Matrix.}
\item{eigenvalues}{A vector containing the estimation of the d+1 main eigenvalues of the variance, where d+1 is the optimal choice belong K.}
\item{MCM_eigenvalues}{A vector containing the estimation of the d+1 main eigenvalues of the Median Covariation Matrix, where d+1 is the optimal choice belong K.}
\item{cap}{The result given for capushe for selecting d if the length of K is larger than 10.}
\item{reduction_results}{A list containing the results for all possible K.}
}
\references{
Cardot, H., Cenac, P. and Zitt, P-A. (2013). Efficient and fast estimation of the geometric median in Hilbert spaces with an averaged stochastic gradient algorithm. \emph{Bernoulli}, 19, 18-43.

Cardot, H. and Godichon-Baggioni, A. (2017). Fast Estimation of the Median Covariation Matrix with Application to Online Robust Principal Components Analysis.  \emph{Test}, 26(3), 461-480

Vardi, Y. and Zhang, C.-H. (2000). The multivariate L1-median and associated data depth. \emph{Proc. Natl. Acad. Sci. USA}, 97(4):1423-1426.

}
\seealso{
See also \code{\link{Robust_Mahalanobis_regression}}, \code{\link{Robust_regression}} and \code{\link{RobRegression-package}}.
}
\examples{
\donttest{
q<-100
d<-10
n<-2000
Sigma<- diag(c(d:1,rep(0,q-d)))+ diag(rep(0.1,q))
X=mvtnorm::rmvnorm(n=n,sigma=Sigma)
RobVar = Robust_Variance(X,K=q)
sum((RobVar$Sigma-Sigma)^2)/q
}
}
