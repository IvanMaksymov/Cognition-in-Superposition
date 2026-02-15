	REAL*8 FUNCTION VEC_ABS(X)
	
	REAL*8 X(3)
	
	VEC_ABS = DSQRT(X(1)**2 + X(2)**2 + X(3)**2)
	
	END




	SUBROUTINE CTIMES(X, Y, RES)
	
	REAL*8 X(3), Y(3), RES(3)
	
	RES(1) = X(2)*Y(3) - X(3)*Y(2)
	RES(2) = X(3)*Y(1) - X(1)*Y(3)
	RES(3) = X(1)*Y(2) - X(2)*Y(1)
	
	END
	


	
	REAL*8 FUNCTION CDOT(X, Y)
	
	REAL*8 X(3), Y(3)
	
	CDOT = X(1)*Y(1) + X(2)*Y(2) + X(3)*Y(3)
	
	END
