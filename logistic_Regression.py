
# coding: utf-8

# In[3]:


import numpy as np
import sys
epoch=int(sys.argv[7])        # position to input the 7th argument
eta=0.5
modelnum=int(sys.argv[8]) #position to input 8th argument  

if(modelnum==1):
    thaap=open(sys.argv[6],"w")
    ar=np.genfromtxt(sys.argv[1],skip_header=0,delimiter='\t',dtype=None,unpack=True) #position to input 1st argument
    ar2=np.genfromtxt(sys.argv[3],skip_header=0,delimiter='\t',dtype=None,unpack=True) #position to input 3rd argument
    ar3=np.genfromtxt(sys.argv[2],skip_header=0,delimiter='\t',dtype=None,unpack=True) #position to input 2nd argument
    da1=len(ar[0])
    # print(ar[0,0])
    # print(ar[1,17])
    lbl=np.unique(ar[1,:])
    ftr=np.unique(ar[0,:])
#     print(len(lbl))
#     print(len(ftr))
    # print(lbl)
    # print(ftr)
    lbdict={}
    ftrdict={}
    # print(desmat.shape)
    for i in range(len(lbl)):
        lbdict[lbl[i]]=i
    for j in range(len(ftr)):
        ftrdict[ftr[j]]=j
    # print(lbdict)
    # print(ftrdict)
    dim1=len(lbl)
#     print(dim1)
    dim2=len(ftr)+1
#     print(dim2)
    theta=np.zeros((dim1,dim2))
    thetadash=np.zeros((dim1,dim2))
    #each epoch
    # print(ftrdict) 

    # calculating log likelihood for some random array
    def nll(arx,threat):
        df=len(arx[0])
        dpool=0.0
        for rt in range(df):
            fre=arx[0,rt]
    #         print(feature)
            lbe=arx[1,rt]
    #         print(label)
            d1=lbdict[lbe]
    #         print(dm1)
            d2=ftrdict[fre]
            dn=0
            for aq in range(dim1):
                dn=dn+np.exp(threat[aq,d2]+threat[aq,-1])
            dpool=dpool+np.log((np.exp(threat[d1,d2]+threat[d1,-1]))/dn)
        return(-dpool/df) 

    #for calculating error of some random array
    def errcal(ard):
        pl=len(ard[0])
        ajooba=[]
        def jugnu(threat,ft):
            prob=np.zeros(dim1)
            for az in range(dim1):
                prob[az]=threat[az,ft]+threat[az,-1]

            return(np.argmax(prob))
        qwe=0
        for ku in range(pl):
            feat=ftrdict[ard[0,ku]]
            lab=lbdict[ard[1,ku]]
            evlab=jugnu(theta,feat)
            ajooba.append(lbl[evlab])
            if(lbl[evlab]==ard[1,ku]):
                qwe=qwe+1
        error=1-(float(qwe)/pl)
        return(error,ajooba)
    for epo in range(epoch):
        deadpool=0.0
        cv=0
        jab=0
        #each example i
        for nl in range(da1):
            feature=ar[0,nl]
    #         print(feature)
            label=ar[1,nl]
    #         print(label)
            dm1=lbdict[label]
    #         print(dm1)
            dm2=ftrdict[feature]
    #         print(dm2)
            deno=0.0
            kns=0.0
            #calculating denominator for each example
            for q in range(dim1):
                deno=deno+np.exp(theta[q,dm2]+theta[q,-1])
    #         print(deno)
    #         print(deadpool)
            #for each k parameter vector, calculating weight to multiply with
            for ty in range(dim1):
                kns=np.exp(theta[ty,dm2]+theta[ty,-1])
                if(ty==dm1):
                    kns=1-(kns/deno)
                else: 
                    kns=-kns/deno
                thetadash[ty,dm2]=theta[ty,dm2]+(kns*eta)
                thetadash[ty,-1]=theta[ty,-1]+(kns*eta)

            theta=thetadash
#         for rt in range(da1):
#             feature=ar[0,rt]
#     #         print(feature)
#             label=ar[1,rt]
#     #         print(label)
#             dm1=lbdict[label]
#     #         print(dm1)
#             dm2=ftrdict[feature]
#     #         print(dm2)
#             deno=0.0
#             kns=0.0

#             #calculating denominator for each example

#             for q in range(dim1):
#                 deno=deno+np.exp(theta[q,dm2]+theta[q,-1])
#             deadpool=deadpool+np.log((np.exp(theta[dm1,dm2]+theta[dm1,-1]))/deno)
        print("epoch={} likelihood(train): {}".format(epo+1,nll(ar,theta)))
        thaap.write("epoch={} likelihood(train): {}".format(epo+1,nll(ar,theta)))
        thaap.write("\n")
        print("epoch={} likelihood(validation): {}".format(epo+1,nll(ar3,theta)))
        thaap.write("epoch={} likelihood(validation): {}".format(epo+1,nll(ar3,theta)))
        thaap.write("\n")
    
    print("error(train): {}".format(errcal(ar)[0]))
    thaap.write("error(train): {}".format(errcal(ar)[0]))
    thaap.write("\n")
    print("error(test): {}".format(errcal(ar2)[0]))
    thaap.write("error(test): {}".format(errcal(ar2)[0]))
    thaap.write("\n")
    thaap.close()
    # ----------------------------------------------------------------------------------------------------------------    
    # building theta for model 2
    #label files created

    armp=open(sys.argv[1], "r").read().splitlines()    # position to input 1st argument
    armp2=open(sys.argv[3], "r").read().splitlines()    # position to input third argument
    def model2(wd,nb):
        wd=np.array(wd)
        cst=len(wd)
        tempar=[]
        for frodo in range(cst):
            if(wd[frodo]==''):
                tempar.append(frodo)
        for flip in tempar:        
            nb=np.insert(nb,flip,"XYZ",axis=1)
        return(nb,tempar)
    # print(model2(ar))
    arfm2=model2(armp,ar)[0]
    indexes1=model2(armp,ar)[1]
    ar2fm2=model2(armp2,ar2)[0]
    indexes2=model2(armp2,ar2)[1]
    # print(indexes1)
    # print(indexes2)
    # print(arfm2)
    # print(ar)
    filtrain=np.array(errcal(ar)[1])
    filtest=np.array(errcal(ar2)[1])
    # print(fil1)
    # print(fil2) 
    def labela_test(labfile,indexd,gulfam):
        pj=0
        for rt in range(len(labfile)):
            pj=pj+1
            if((pj-1) in indexd):
                gulfam.write("\n")
                pj=pj+1 
            gulfam.write(labfile[rt])
            gulfam.write("\n")
        gulfam.close()
        return(gulfam)


    labprint1=open(sys.argv[4],"w")  # position for the argument 4th
    labprint2=open(sys.argv[5],"w")   # position for the argument 5th
    labela_test(filtrain,indexes1,labprint1)
    labela_test(filtest,indexes2,labprint2)
#--------------------------------second theta-----------------------------------------------------------------------
if(modelnum==2):
    thaap=open(sys.argv[6],"w")
    ar=np.genfromtxt(sys.argv[1],skip_header=0,delimiter='\t',dtype=None,unpack=True) #position to input 1st argument
    ar2=np.genfromtxt(sys.argv[3]  ,skip_header=0,delimiter='\t',dtype=None,unpack=True) #position to input 3rd argument
    ar3=np.genfromtxt(sys.argv[2],skip_header=0,delimiter='\t',dtype=None,unpack=True) #position2nd argument
    armp=open(sys.argv[1], "r").read().splitlines()    # position to input 1st argument
    armp2=open(sys.argv[3], "r").read().splitlines()    # position to input third argument
    armp3=open(sys.argv[2], "r").read().splitlines()
    arlgt=len(ar[0])
    def model2(wd,nb):
        wd=np.array(wd)
        cst=len(wd)
        tempar=[]
        for frodo in range(cst):
            if(wd[frodo]==''):
                tempar.append(frodo)
        for flip in tempar:        
            nb=np.insert(nb,flip,"XYZ",axis=1)
        return(nb,tempar)
    # print(model2(ar))
    arfm2=model2(armp,ar)[0]
    indexes1=model2(armp,ar)[1]
    ar2fm2=model2(armp2,ar2)[0]
    indexes2=model2(armp2,ar2)[1]
    ar3fm2=model2(armp3,ar3)[0]
    indexes3=model2(armp3,ar3)[1]
    da1=len(arfm2[0])
    lbl=np.unique(ar[1,:])
    
    ftr=np.unique(ar[0,:])
    lbdict={}
    ftrdict={}
    for i in range(len(lbl)):
        lbdict[lbl[i]]=i
    for j in range(len(ftr)):
        ftrdict[ftr[j]]=j
    dim1=len(lbl)
#     print(dim1)
    dim22=(3*(len(ftr)+2)+1)
    dim2=len(ftr)
    theta=np.zeros((dim1,dim22))
    thetadash=np.zeros((dim1,dim22))
    #calculating some random negative log likelihood
    def nll(arx,threat,dada):
        essex=0
        df=len(arx[0])
#         dada=len(ar[0])
        dpool=0.0
        for rt in range(df):
            fre=arx[0,rt]
    #         print(feature)
            lbe=arx[1,rt]
    #         print(label)
            if(fre=="XYZ"):
                continue
            if(arx[0,rt-1]=="XYZ"):
                
                d1=lbdict[lbe]
                d2=ftrdict[fre]+(dim2+2)
                d2p=ftrdict[arx[0,rt+1]]+(2*(dim2+2))
                d2n=dim2
                
            elif(arx[0,rt+1]=="XYZ"):
                
                d1=lbdict[lbe]
                d2=ftrdict[fre]+(dim2+2)
                d2n=ftrdict[arx[0,rt-1]]
                d2p=3*(dim2+2)-1
            else:
                d1=lbdict[lbe]
                d2=ftrdict[fre]+(dim2+2)
                d2n=ftrdict[arx[0,rt-1]]
                d2p=ftrdict[arx[0,rt+1]]+(2*(dim2+2))
            dn=0
            for aq in range(dim1):
                dn=dn+np.exp(threat[aq,d2]+threat[aq,-1]+threat[aq,d2p]+threat[aq,d2n])
            dpool=dpool+np.log((np.exp(threat[d1,d2]+threat[d1,-1]+threat[d1,d2p]+threat[d1,d2n]))/dn)
        return(-dpool/dada) 

    #for calculating error of some random array
    def errcal(ard):
        pl=len(ard[0])
        ajooba=[]
        gaba=0
        def jugnu(threat,ft,ui,io):
            prob=np.zeros(dim1)
            for az in range(dim1):
                prob[az]=threat[az,ft]+threat[az,-1]+threat[az,ui]+threat[az,io]

            return(np.argmax(prob))
        qwe=0
        for ku in range(pl):
            if(ard[1,ku]=="XYZ"):
                continue
            gaba=gaba+1
            feat=ftrdict[ard[0,ku]]
            lab=lbdict[ard[1,ku]]
            if(ard[1,ku+1]=="XYZ"):
                evlab=jugnu(theta,feat+(dim2+2),ftrdict[ard[0,ku-1]],3*(dim2+2)-1)
            elif(ard[1,ku-1]=="XYZ"):
                evlab=jugnu(theta,feat+(dim2+2),dim2,ftrdict[ard[0,ku+1]]+2*(dim2+2))
            else:
                evlab=jugnu(theta,feat+(dim2+2),ftrdict[ard[0,ku-1]],ftrdict[ard[0,ku+1]]+2*(dim2+2))
            ajooba.append(lbl[evlab])
            if(lbl[evlab]==ard[1,ku]):
                qwe=qwe+1
        error=1-(float(qwe)/gaba)
        return(error,ajooba)
    bkup1=arfm2[0]
    bkup2=arfm2[1]
    bkup2=np.append(bkup2,"XYZ")
    bkup2=np.insert(bkup2,0,"XYZ")
    bkup1=np.insert(bkup1,0,"XYZ")
    bkup1=np.append(bkup1,"XYZ")
    arfm2=np.vstack((bkup1,bkup2))
    
    ckup1=ar2fm2[0]
    ckup2=ar2fm2[1]
    ckup2=np.append(ckup2,"XYZ")
    ckup2=np.insert(ckup2,0,"XYZ")
    ckup1=np.insert(ckup1,0,"XYZ")
    ckup1=np.append(ckup1,"XYZ")
    ar2fm2=np.vstack((ckup1,ckup2))
    
    dkup1=ar3fm2[0]
    dkup2=ar3fm2[1]
    dkup2=np.append(dkup2,"XYZ")
    dkup2=np.insert(dkup2,0,"XYZ")
    dkup1=np.insert(dkup1,0,"XYZ")
    dkup1=np.append(dkup1,"XYZ")
    ar3fm2=np.vstack((dkup1,dkup2))
    for epo in range(epoch):
        deadpool=0.0
        cv=0
        jab=0
        cde=0
        #each example i which is nl here
        #theta - 1(.....BOS EOS-bogus)...2(......BOS-b EOS-b).....3(.....BOS-b EOS) bias
        for nl in range(da1+1):
            feature=arfm2[0,nl]
            label=arfm2[1,nl]
            if(label=="XYZ"):
                continue
            if(arfm2[0,nl-1]=="XYZ"):
                dm1=lbdict[label]               
                dm2=ftrdict[feature]+(dim2+2)
                dm2p=ftrdict[arfm2[0,nl+1]]+(2*(dim2+2))
                dm2n=dim2
            elif(arfm2[0,nl+1]=="XYZ"):
                dm1=lbdict[label]
                dm2=ftrdict[feature]+(dim2+2)
                dm2n=ftrdict[arfm2[0,nl-1]]
                dm2p=3*(dim2+2)-1
            else:
                dm1=lbdict[label]
                dm2=ftrdict[feature]+(dim2+2)
                dm2p=ftrdict[arfm2[0,nl+1]]+2*(dim2+2)
                dm2n=ftrdict[arfm2[0,nl-1]]
            deno=0.0
            kns=0.0
            for q in range(dim1):
                deno=deno+np.exp(theta[q,dm2]+theta[q,dm2n]+theta[q,-1]+theta[q,dm2p])
            #for each k parameter vector, calculating weight to multiply with
            for ty in range(dim1):
                kns=np.exp(theta[ty,dm2]+theta[ty,-1]+theta[ty,dm2p]+theta[ty,dm2n])
                if(ty==dm1):
                    kns=1-(kns/deno)
                else: 
                    kns=-kns/deno
                thetadash[ty,dm2]=theta[ty,dm2]+(kns*eta)
                thetadash[ty,-1]=theta[ty,-1]+(kns*eta)
                thetadash[ty,dm2p]=theta[ty,dm2p]+(kns*eta)
                thetadash[ty,dm2n]=theta[ty,dm2n]+(kns*eta)
            theta=thetadash
        print("epoch={} likelihood(train): {}".format(epo+1,nll(arfm2,theta,len(ar[0]))))
        thaap.write("epoch={} likelihood(train): {}".format(epo+1,nll(arfm2,theta,len(ar[0]))))
        thaap.write("\n")
        print("epoch={} likelihood(validation): {}".format(epo+1,nll(ar3fm2,theta,len(ar3[0])))) 
        thaap.write("epoch={} likelihood(validation): {}".format(epo+1,nll(ar3fm2,theta,len(ar3[0]))))
        thaap.write("\n")
    
    print("error(train): {}".format(errcal(arfm2)[0]))
    thaap.write("error(train): {}".format(errcal(arfm2)[0]))
    thaap.write("\n")
    print("error(test): {}".format(errcal(ar2fm2)[0]))
    thaap.write("error(test): {}".format(errcal(ar2fm2)[0]))
    thaap.write("\n")
    thaap.close()




    armp=open(sys.argv[1], "r").read().splitlines()    # position to input 1st argument
    armp2=open(sys.argv[3] , "r").read().splitlines()    # position to input third argument
    def model2(wd,nb):
        wd=np.array(wd)
        cst=len(wd)
        tempar=[]
        for frodo in range(cst):
            if(wd[frodo]==''):
                tempar.append(frodo)
        for flip in tempar:        
            nb=np.insert(nb,flip,"XYZ",axis=1)
        return(nb,tempar)
    # print(model2(ar))
#     arfm2=model2(armp,ar)[0]
    indexes1=model2(armp,ar)[1]
#     ar2fm2=model2(armp2,ar2)[0]
    indexes2=model2(armp2,ar2)[1]
    # print(indexes1)
    # print(indexes2)
    # print(arfm2)
    # print(ar)
    filtrain=np.array(errcal(arfm2)[1])
    filtest=np.array(errcal(ar2fm2)[1])
    # print(fil1)
    # print(fil2) 
    def labela_test(labfile,indexd,gulfam):
        pj=0
        for rt in range(len(labfile)):
            pj=pj+1
            if((pj-1) in indexd):
                gulfam.write("\n")
                pj=pj+1 
            gulfam.write(labfile[rt])
            gulfam.write("\n")
        gulfam.close()
        return(gulfam)


    labprint1=open(sys.argv[4],"w")  # position for the argument 4th
    labprint2=open(sys.argv[5],"w")   # position for the argument 5th
    labela_test(filtrain,indexes1,labprint1)
    labela_test(filtest,indexes2,labprint2)

