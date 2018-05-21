function u= ACM_LPF(u,nu,timestep,mu,epsilon,lambda1,lambda2,e1,e2)

    u=NeumannBoundCond(u);
    K=curvature_central(u);   
    
    DrcU=(epsilon/pi)./(epsilon^2.+u.^2);  
    H=0.5*(1+(2/pi)*atan(u./epsilon));  
    
    LPFterm=-DrcU.*(e1.*lambda1-e2.*lambda2);
    debug=del2(u);
    PenaltyTerm=mu*(4*del2(u)-K);                              
    LengthTerm=nu.*DrcU.*K;    
    u=u+timestep*(LengthTerm+PenaltyTerm+LPFterm);  %LIFterm

                                                                                                               
function g = NeumannBoundCond(f)
% Neumann boundary condition
[nrow,ncol] = size(f);
g = f;


% %debug
% x=[1:(100*100)]
% x=reshape(x,100,100)
% [xnrow,xncol] = size(x);
% x([1 xnrow],[1 xncol]) = x([3 xnrow-2],[3 xncol-2]);
% x([1 xnrow],2:end-1) = x([3 xnrow-2],2:end-1); 
% x(2:end-1,[1 ncol]) = x(2:end-1,[3 xncol-2]);  
% %


g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  

function k = curvature_central(u)                       
% compute curvature
[ux,uy] = gradient(u);                                  
normDu = sqrt(ux.^2+uy.^2+1e-10);                                                                            
Nx = ux./normDu;                                       
Ny = uy./normDu;
[nxx,~] = gradient(Nx);                              
[~,nyy] = gradient(Ny);                              
k = nxx+nyy;            
