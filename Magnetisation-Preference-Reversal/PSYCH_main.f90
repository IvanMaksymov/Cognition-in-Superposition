	PROGRAM FDTD !big
    
    IMPLICIT NONE
    
    INTEGER MAX_ITER, NY
    INTEGER STIMULUS_SIZE
    !PARAMETER (MAX_ITER =  2D7)
    PARAMETER (NY = 20)
    REAL*8 PI
    REAL*8 LIGHT_SPEED
    REAL*8 MU_0
    REAL*8 EPSILON_0
    INTEGER J, JB1, JB2, J1, J2, K, I, IH0, J_SOURCE, JMPY
!    INTEGER ITERATION
    REAL*8 STIMULUS
    REAL*8 CURRENTSIMULATEDTIME
    REAL*8 TOTALSIMULATEDTIME
    REAL*8 OMEGA
    REAL*8 LAMBDA
    REAL*8 DY
    REAL*8 DT
    REAL*8 TW, T0
    REAL*8 DTDY
    
    REAL*8 EPSILON1(1:NY+1), EPS_INF, SGM(1:NY+1)
    REAL*8 EZ(1:NY+1), JZ(1:NY+1)
	REAL*8 HX(1:NY), HY(1:NY-1) !step n+1/2
	REAL*8 HXN(1:NY), HYN(1:NY-1) !step n
	REAL*8 oldHXN(1:NY) !step n-1/2		
	REAL*8 BX(1:NY) !step n+1/2
	REAL*8 oldBX(1:NY) !step n-1/2	
	REAL*8 BXN(1:NY) !step n
	
	REAL*8 MX(1:NY+1), MY(1:NY+1), MZ(1:NY+1) !magnetization step n+1/2
	REAL*8 oldMX(1:NY+1), oldMY(1:NY+1), oldMZ(1:NY+1) !magnetization step n+1/2
	REAL*8 MXN(1:NY+1), MYN(1:NY+1), MZN(1:NY+1) !magnetization step n
	REAL*8 HEX_X(1:NY+1), HEX_Y(1:NY+1), HEX_Z(1:NY+1) !excahnge field
	REAL*8 oldMXN(1:NY+1), oldMYN(1:NY+1), oldMZN(1:NY+1) !magnetization step n, previous iteration

	REAL*8 DFT_EZ1(1:NY+1), DFT_JZ1(1:NY+1)
	REAL*8 DFT_HX1(1:NY)
	REAL*8 DFT_MX1(1:NY+1), DFT_MY1(1:NY+1) 
	REAL*8 DFT_HEX_X1(1:NY+1), DFT_HEX_Y1(1:NY+1)

	REAL*8 DFT_EZ2(1:NY+1), DFT_JZ2(1:NY+1)
	REAL*8 DFT_HX2(1:NY)
	REAL*8 DFT_MX2(1:NY+1), DFT_MY2(1:NY+1) 
	REAL*8 DFT_HEX_X2(1:NY+1), DFT_HEX_Y2(1:NY+1)
	
	REAL*8 SUM_MX(1:NY+1), SUM_MY(1:NY+1), SUM_HX(1:NY)
		    
    INTEGER PLOT_MODULUS, DUMMY
    REAL*8 WORK_FREQ, FT_FREQ(3)
    REAL*8 R, W_CU, W_PY, D, MS1, MS2, A1, A2, A12, H0, ALPHA1, ALPHA2, GAMMA1, GAMMA2, SIGMA1, SIGMA2
    REAL*8 MS, A, ALPHA, GAMMA
    REAL*8 OE, GAUSS, ERG_over_CM, TESLA, ERG_over_CM2
    REAL*8 AZ, X0_X, X0_Y, ACC
	
    INTEGER CENTER_Y
    
    REAL*8 C1,C2,TMP1,TMP2,TMP,WINDOW,A_VEC(3),X_VEC(3),A_TIMES_X_VEC(3),M0_VEC(3),H0_VEC(3),HEFF(3),SUM_VEC(3),TMP_VEC(3)
    REAL*8 M_VEC(3), A_M_VEC(3), A_M_DOT
    REAL*8 VEC_ABS, CDOT, U1, U2, U3, U4, U5, U6, U7
    INTEGER FLAG, FLAG_2_FILMS, NH0, ITMP
    REAL*8 T1, T2, TA_LAYER
    REAL*8 SIGMA0, E, G, MU_B, Is, Is0, Id, B, E_VEC(3), ZETA(NY), modM
    INTEGER CNT
    REAL*8 K_VEC, H01
    
    integer, parameter :: verylong = selected_int_kind(38)
	integer(verylong) :: CUR_STEP
    
! CPML PARAMETERS
	REAL*8 M
! POLINOMIAL GRADING
	REAL*8 D2Y
! PML THICKNESS IN METERS
	REAL*8 SIGMA_MAX
! MAX VALUE OF SIGMA PARAMETER
	REAL*8 A_MAX
! MAX VALUE OF A PARAMETER
	REAL*8 KMAX
! MAX VALUE OF K PARAMETER
    
    OPEN (UNIT=54,FILE='DETECTOR5.OUT',STATUS='UNKNOWN')
    
    PLOT_MODULUS = 1D5
    EPS_INF = 1.0D0
    CENTER_Y = NY/2
    
    ACC = 1.0D-6
    
    PI = 3.1415926D0
    LIGHT_SPEED = 299792458.0D0
    MU_0 = 1.256637D-6
    EPSILON_0 =  8.8541878D-12
    
    OE = 1.0D3/4.0D0/PI
	GAUSS = 1.0D3
	ERG_over_CM = 1.0D-7/1.0D-2
	ERG_over_CM2 = 1.0D-7/1.0D-4
	TESLA = 795.744D3
    MU_B = 9.274009994D-24
    E = 1.602176634D-19
    G = 2.0

    W_PY = 2.5D0*4.0D-9

    TOTALSIMULATEDTIME = 10.0D-9!50.0D-9
    
    NH0 = 0
    
    DO IH0 = 1, NH0+1
    
!    call cpu_time(t1)
    
	OPEN(UNIT=22,FILE='KVEC.IN',STATUS='OLD')
	READ(22,*) K_VEC
	CLOSE(22)  
	
	OPEN(UNIT=22,FILE='H0.IN',STATUS='OLD')
	READ(22,*) H01
	CLOSE(22)  
	
	!WRITE(*,*) 'K: (m^-1) = ', K_VEC      
    
    
    !H0 = 4000.0D0*OE*(1.0D0 - DBLE(IH0-1)/DBLE(NH0))
    H0 = H01*OE
    Is0 = K_VEC*1.0D-4!0.05D-4!0.3785D-4
    WRITE(*,*) 'H0: (A/m, Oe) = ', H0, H0/OE      
    WRITE(*,*) 'Is0: (A) = ', Is0      
	    
    DO J = 1,NY+1
    
        EPSILON1(J) = EPS_INF
        HEX_X(J) = 0.0d0
        HEX_Y(J) = 0.0d0        
        SGM(J) = 0.0d0
        EZ(J) = 0.0d0
        JZ(J) = 0.0d0		
		MX(J) = 0.0d0
		MY(J) = 0.0d0
		MZ(J) = 0.0d0
		MXN(J) = 0.0d0
		MYN(J) = 0.0d0
		MZN(J) = 0.0d0
		HEX_X(J) = 0.0d0
		HEX_Y(J) = 0.0d0
		HEX_Z(J) = 0.0d0
		oldMXN(J) = 0.0d0
		oldMYN(J) = 0.0d0
		oldMZN(J) = 0.0d0
		SUM_MX(J) = 0.0d0
		SUM_MY(J) = 0.0d0
    
    ENDDO

    DO J = 1,NY    
    
		HX(J) = 0.0d0
		HXN(J) = 0.0d0
		BX(J) = 0.0d0
		oldBX(J) = 0.0d0
		oldHXN(J) = 0.0d0
		BXN(J) = 0.0d0
				
		DFT_HX1(J) = 0.0D0
		DFT_HX2(J) = 0.0D0
					
		DFT_MX1(J) = 0.0D0
		DFT_MX2(J) = 0.0D0

		DFT_MY1(J) = 0.0D0
		DFT_MY2(J) = 0.0D0

		DFT_EZ1(J) = 0.0D0
		DFT_EZ2(J) = 0.0D0

		DFT_JZ1(J) = 0.0D0
		DFT_JZ2(J) = 0.0D0
			
		DFT_HEX_X1(J) = 0.0D0
		DFT_HEX_X2(J) = 0.0D0

		DFT_HEX_Y1(J) = 0.0D0
		DFT_HEX_Y2(J) = 0.0D0
					
	ENDDO
	
	DO J = 1,NY - 1 
	
		HYN(J) = 0.0D0
		HY(J) = 0.0D0
	
	ENDDO

    DY = 1.0D-9!1.6667D-9
    DT = 1.0D0*DY*DSQRT(EPS_INF)/LIGHT_SPEED
    PRINT 5, DY, DT
    5 FORMAT (1X,' DY=',E10.4,' DT=',E10.4)
    
!    MAX_ITER = INT(10.0D-9/DT)
!    WRITE(*,*) 'Total number of iterations = ', MAX_ITER

	D = 0.0D0 !pinning parameter
	MS1 = 8042.0D0*GAUSS/4.0D0/PI
	A1 = 0.55D-6*ERG_over_CM
	A2 = A1
	A12 = A1/DY
	ALPHA1 = 0.008
	SIGMA1 = 4.5D6
	GAMMA1 = 2.0D0*PI*2.92D6/OE 
	
	WRITE(*,*) 'Ms1: (A/m, Gauss) = ', MS1, MS1*4.0D0*PI/GAUSS
	WRITE(*,*) 'A1: (J/m, erg/cm) = ', A1, A1/ERG_over_CM	
	WRITE(*,*) 'A12: (J/m^2, erg/cm^2) = ', A12, A12/ERG_over_CM2
	WRITE(*,*) 'Sigma1: (S/m) = ', SIGMA1	
	WRITE(*,*) 'Alpha1:  = ', ALPHA1	
	WRITE(*,*) 'Gamma1: (Hz m/A, MHz/Oe/2pi) = ', GAMMA1, GAMMA1*OE/1D6/2.0D0/PI
	    		
	CALL CONFIG(NY,EPSILON1,SGM,0,DY,W_CU,W_PY,JB1,JB2,J1,J2,SIGMA1,SIGMA2,EPS_INF)
	
	WRITE(*,*) 'Py LAYER', W_PY

    JMPY = CENTER_Y+1

    WRITE(*,*) 'CENTRE OF THE FREE Py LAYER', JMPY

	!INITIAL CONDITION FOR MAGNETIZATION
	!FREE LAYER
	DO J=J1, JB1-1
		
		MX(J) = 0.0D0
		MY(J) = 0.0D0
		MZ(J) = MS1

		MXN(J) = 0.0D0
		MYN(J) = 0.0D0
		MZN(J) = MS1
				
	ENDDO

   	E_VEC(1) = 1.0D0*DCOS(PI/2.0D0-PI/180000.0D0)	
    E_VEC(2) = 0.0D0
    E_VEC(3) = 1.0D0!*DSIN(PI/2.0D0-PI/180000.0D0)

    CURRENTSIMULATEDTIME = -DT
    Is = Is0
 
 	CUR_STEP = -1
 	CNT = 0
 
    ! MAIN LOOP    
    DO WHILE (CURRENTSIMULATEDTIME.LE.TOTALSIMULATEDTIME) !ITERATION = 0, MAX_ITER

        CURRENTSIMULATEDTIME = CURRENTSIMULATEDTIME + DT
		
		CUR_STEP=CUR_STEP+1!NINT(CURRENTSIMULATEDTIME/DT)

		IF (CUR_STEP.EQ.PLOT_MODULUS) THEN

!            modM = DSQRT(MX(JMPY)**2+MY(JMPY)**2+MZ(JMPY)**2)
    
!			PRINT*, 'STEP # ', CUR_STEP
			PRINT*, 'TIME, NANO-SECONDS ', CURRENTSIMULATEDTIME/1.0D-9
!			PRINT*, 'STEP # ', ITERATION, ' OUT OF', MAX_ITER, ' TIME, NANO-SECONDS ', CURRENTSIMULATEDTIME/1.0D-9
!			PRINT*, 'TIME, NANO-SECONDS ', CURRENTSIMULATEDTIME/1.0D-9
!            PRINT*, 'Pulse ', iPULSE
            WRITE(54,'(7(1X,E15.8))') CURRENTSIMULATEDTIME, MX(JMPY), MY(JMPY), MZ(JMPY), EZ(JMPY)*SGM(JMPY), Is
!            WRITE(*,'(7(1X,E15.8))') CURRENTSIMULATEDTIME, MX(JMPY), MY(JMPY), MZ(JMPY), EZ(JMPY)*SGM(JMPY), Is
!            WRITE(*,*) CURRENTSIMULATEDTIME/1e-9, MX(JMPY), MY(JMPY), MZ(JMPY), Is!ALPHA1*(1.0D0 + 10.0D3*ZETA(JMPY) + 10.0D3*ZETA(JMPY)**2)
			
!			call cpu_time(t2)
!			print *,'CPU time in seconds ',t2-t1

			CUR_STEP = 0
			
		ENDIF
                
        ! UPDATE THE BX VALUES -- STEP 1

		DO J=1, NY-1

			DTDY=DT/DY
			oldBX(J) = BX(J)! step n-1/2
			BX(J) = BX(J) + ( - DTDY*(EZ(J+1) - EZ(J))) !step n+1/2
			BXN(J) = 0.5D0*(BX(J) + oldBX(J)) !step n
			oldHXN(J) = HXN(J)
			
		ENDDO

		! STEP 2. No spatial interpolation of BXN for 1D Model because bx(n-1/2) is already at the same point as m

		oldMX = MX !previous iteration
		oldMY = MY !previous iteration
		oldMZ = MZ !previous iteration

		! STEP 3
		K = 0
		DO !iterations
			!inside the free layer only
			DO J=J1, JB1-1

		        A = A1
		        GAMMA = GAMMA1
		        ALPHA = ALPHA1*(1.0D0 + 1.0D3*ZETA(J) + 1.0D3*ZETA(J)**2)
		        !ALPHA = ALPHA1*(1.0D0 + 10.0D3*ZETA(J) + 10.D3*ZETA(J)**2)
											
				HEFF(1) = HXN(J) + HEX_X(J)
				HEFF(2) = HYN(J) + HEX_Y(J)
				HEFF(3) = H0 + HEX_Z(J)

                modM = DSQRT(MXN(J)**2 + MYN(J)**2 + MZN(J)**2)

                SIGMA0 = 1.0*G*MU_B/2.0D0/E/modM/W_PY/DY**2    
                B = SIGMA0*Is*DT/2.0D0/modM !COEFFICIENT

				A_VEC(1) = -(0.5D0*DABS(GAMMA)*DT*HEFF(1) + ALPHA*MX(J)/modM - B*(MYN(J)*E_VEC(3)-E_VEC(2)*MZN(J)))
				A_VEC(2) = -(0.5D0*DABS(GAMMA)*DT*HEFF(2) + ALPHA*MY(J)/modM + B*(MXN(J)*E_VEC(3)-E_VEC(1)*MZN(J))) 
				A_VEC(3) = -(0.5D0*DABS(GAMMA)*DT*HEFF(3) + ALPHA*MZ(J)/modM - B*(MXN(J)*E_VEC(2)-E_VEC(1)*MYN(J)))
												
				oldMXN(J) = MXN(J) !previous iteration
				oldMYN(J) = MYN(J) !previous iteration
				oldMZN(J) = MZN(J) !previous iteration
				
				M_VEC(1) = MX(J)
				M_VEC(2) = MY(J)
				M_VEC(3) = MZ(J)
				
				CALL CTIMES(A_VEC, M_VEC, A_M_VEC)
				
				MXN(J) = (M_VEC(1) + CDOT(A_VEC, M_VEC)*A_VEC(1) - A_M_VEC(1))/(1.0D0 + VEC_ABS(A_VEC)**2)
				MYN(J) = (M_VEC(2) + CDOT(A_VEC, M_VEC)*A_VEC(2) - A_M_VEC(2))/(1.0D0 + VEC_ABS(A_VEC)**2)
				MZN(J) = (M_VEC(3) + CDOT(A_VEC, M_VEC)*A_VEC(3) - A_M_VEC(3))/(1.0D0 + VEC_ABS(A_VEC)**2)
							
			ENDDO
			
			!BOUNDARY CONDITION FOR MXN and MYN
			
			!Outer surface of free layer 1
			MYN(J1) = MYN(J1+1)/(1.0D0 - D*DY)
			MXN(J1) = MXN(J1+1)
			MZN(J1) = MZN(J1+1)/(1.0D0 - D*DY) !????????????????????????????

			!Outer surface of free layer 2
			MYN(JB1-1) = MYN((JB1-1)-1)/(1.0D0 - D*DY)
			MXN(JB1-1) = MXN((JB1-1)-1)
			MZN(JB1-1) = MZN((JB1-1)-1)/(1.0D0 - D*DY) !????????????????????????????

            !!!! For the FIXED LAYER the BC are self-satisfied
									
			!EXCHANGE FIELD FREE LAYER
			DO J=J1+1, (JB1-1)-1
			
				A = A1
                modM = DSQRT(MXN(J)**2 + MYN(J)**2 + MZN(J)**2)    
									
				HEX_X(J) = (2.0D0*A/(MU_0*modM**2))*(MXN(J+1)-2.0D0*MXN(J)+MXN(J-1))/DY**2
				HEX_Y(J) = (2.0D0*A/(MU_0*modM**2))*(MYN(J+1)-2.0D0*MYN(J)+MYN(J-1))/DY**2
				HEX_Z(J) = (2.0D0*A/(MU_0*modM**2))*(MZN(J+1)-2.0D0*MZN(J)+MZN(J-1))/DY**2   				
				
			ENDDO
			
			! EXTRAPOLATION TO FIND HEX IN J1
			HEX_X(J1) = 2.0D0*HEX_X(J1+1)-HEX_X(J1+2)
			HEX_Y(J1) = 2.0D0*HEX_Y(J1+1)-HEX_Y(J1+2)
			HEX_Z(J1) = 2.0D0*HEX_Z(J1+1)-HEX_Z(J1+2)
										
			! EXTRAPOLATION TO FIND HEX IN JB1
			HEX_X(JB1-1) = 2.0D0*HEX_X((JB1-1)-1)-HEX_X((JB1-1)-2)
			HEX_Y(JB1-1) = 2.0D0*HEX_Y((JB1-1)-1)-HEX_Y((JB1-1)-2)
			HEX_Z(JB1-1) = 2.0D0*HEX_Z((JB1-1)-1)-HEX_Z((JB1-1)-2)			
							
			!MAGNETIC FIELD X step n
			DO J=1, NY			
				
				HXN(J) = BXN(J)/MU_0 - MXN(J) !step n, current iteration
				
			ENDDO
			
			!MAGNETIC FIELD Y step n
 
			DO J=1, NY-1

				!!!-------->HYN(J) = - MYN(J) WAS WRONG!!!
				HYN(J) =  - 0.5D0*(MYN(J+1)+MYN(J))

			ENDDO
						
			TMP1 = 0.0D0
			TMP2 = 0.0D0
		
			DO J=J1, JB1-1
			
				TMP1 = TMP1 + DSQRT(MXN(J)**2+MYN(J)**2+MZN(J)**2)
				TMP2 = TMP2 + DSQRT(oldMXN(J)**2+oldMYN(J)**2+oldMZN(J)**2)	
				
			ENDDO						
						
			IF(TMP1/TMP2-1.0D0.GT.ACC) THEN

				FLAG = 1
					
			ELSE
				
				FLAG = 0
										
			ENDIF			
			
			K = K + 1
			
			IF(FLAG.EQ.0) EXIT
						
		ENDDO!WHILE
		
		IF(K.GT.100) THEN
			WRITE(*,*) '#ITERS DONE = ', K
			WRITE(*,*) 'TOO MANY ITERATIONS! SOMETHING IS WRONG! STOP'
			STOP 				
		ENDIF					

		! Step 5
33		DO J=J1, JB1-1

			MX(J) = 2.0D0*MXN(J) - MX(J) !step n+1/2
			MY(J) = 2.0D0*MYN(J) - MY(J) !step n+1/2
			MZ(J) = 2.0D0*MZN(J) - MZ(J) !step n+1/2

            ZETA(J) = ((MX(J)-oldMX(J))/DT)**2 + ((MY(J)-oldMY(J))/DT)**2 + ((MZ(J)-oldMZ(J))/DT)**2
            modM = DSQRT(MX(J)**2+MY(J)**2+MZ(J)**2)
            ZETA(J) = ZETA(J)/(4.0D0*PI*GAMMA*modM)**2/modM**2
			
		ENDDO	
				
		!Step 6: interpolation is not required for the 1D model

		DO J=1, NY-1

! UPDATE THE HX VALUES
			
			HX(J) = BX(J)/MU_0 - MX(J)			
			
!	UPDATE THE HY VALUES

			HY(J) = -0.5D0*(MY(J+1)+MY(J))

		ENDDO

		! EXCITATION
        ! Removed
    		
! UPDATE THE EZ VALUES

		DO J=2, NY

         	C1=2.0D0*EPSILON1(J)*EPSILON_0-SGM(J)*DT
    		C2=2.0D0*EPSILON1(J)*EPSILON_0+SGM(J)*DT

			EZ(J) = (C1/C2)*EZ(J)+ (2.0D0*DT/C2)*(- (HX(J) - HX(J-1))/DY - JZ(J))		
	
		ENDDO
    
    ENDDO!FDTD

	ENDDO!H0
    
    CLOSE(54)
    
END
    
    
    

