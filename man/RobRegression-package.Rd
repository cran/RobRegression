\name{RobRegression-package}
\alias{RobRegression-package}
%\alias{Gmedian}
\docType{package}
\title{
\packageTitle{RobRegression}
}

\description{
This Package focuses on multivariate robust Guassian linear regression.
We provide a function \code{\link{Robust_Mahalanobis_regression}} which enables to obtain robust estimates of the parameters of Multivariate Gaussian Linear Models with the help of the Mahalanobis distance, using a Stochastic Gradient algorithm or a Fix point. This is based on the function \code{\link{Robust_Variance}} which allows to obtain robust estimation of the variance, and so, also for low rank matrices (see Godichon-Baggioni and RObin (2024) <doi:10.1007/s11222-023-10362-9>)
\packageDescription{RobRegression}.
}

\details{
\packageDESCRIPTION{RobRegression}
\packageIndices{RobRegression}
}

\author{
\packageAuthor{RobRegression}

Maintainer: \packageMaintainer{RobRegression}
}

\references{
Cardot, H., Cenac, P. and Zitt, P-A. (2013). Efficient and fast estimation of the geometric median in Hilbert spaces with an averaged stochastic gradient algorithm. \emph{Bernoulli}, 19, 18-43.

Cardot, H. and Godichon-Baggioni, A. (2017). Fast Estimation of the Median Covariation Matrix with Application to Online Robust Principal Components Analysis. \emph{Test}, 26(3), 461-480

Godichon-Baggioni, A. and Robin, S. (2024). Recursive ridge regression using second-order stochastic \emph{algorithms. Computational Statistics & Data Analysis, 190, 107854.}

Vardi, Y. and Zhang, C.-H. (2000). The multivariate L1-median and associated data depth. \emph{Proc. Natl. Acad. Sci. USA}, 97(4):1423-1426.
}

