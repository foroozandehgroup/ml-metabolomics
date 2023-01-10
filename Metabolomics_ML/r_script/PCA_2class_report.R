params <-
list(user = "your name", filename = "test_data", scale = "standard", 
    q = 0.95, control_level = "RRMS", case_level = "SPMS", col_control = "green", 
    col_case = "blue")

## ----setup , include=FALSE---------------------------------------------------
#colours for plots control first then case
colours<-c(params$col_control,params$col_case)
#set transparency
transparent <- adjustcolor(colours,alpha.f=0.5)

knitr::opts_chunk$set(echo = FALSE, 
                      fig.align = "center",
                      fig.width = 8,
                      fig.height = 8,
                      dev = "png",
                      cache = FALSE,
                      error = TRUE,
                      tidy = TRUE,
                      palette(transparent))


## ----library, include=FALSE, error=TRUE--------------------------------------
library(vegan)
library(scatterplot3d)
library(ropls)
library(tcltk)
library(caret)
library(ggsignif)
library(ggrepel)
library(knitr)
library(tools)
library(ggplot2)
library(gridExtra)
library(knitr)


## ----read_data, include=TRUE-------------------------------------------------
Data<-read.csv(paste(params$filename,".csv",sep=""), head=T)
Data$Class<-as.factor(Data$Class)
control<-params$control_level
case<-params$case_level

if(file.exists(paste(params$filename,".csv",sep=""))==FALSE){
  warn1<-"WARNING:no data file in working directory with this name"
  message(warn1)
} else
  warn1<-"Data file (.csv) found SUCCESSFULLY"

if(control %in% levels(Data$Class)==FALSE){
  warn2<-"WARNING: Control name not present in data file Class column"
  message(warn2)
} else
  warn2<-"control group found in Class column SUCCESSFULLY"

if(case %in% levels(Data$Class)==FALSE){
  warn3<-"WARNING: Case name not present in data file Class column"
  message(warn3)
} else
  warn3<-"case group found in Class column SUCCESSFULLY"

if(colours[1]==colours[2]){
  warn4<-"WARNING: You have selected the same colour for control and case groups, plots may not be very informative!"
  message(warn4)
} else
  warn4<-"Different colours selected for case and control groups. Good idea!"


pca.plot.cap<-paste("PCA scores plot. ", control, " (",colours[1]," circles, n = ", n.control,"), ",case, " (",colours[2]," squares, n = ", n.case,").", sep="")

pca.hotel.cap<-paste("PCA scores plot. ", control, " (",colours[1]," circles, n = ", n.control,"), ",case, " (",colours[2]," squares, n = ", n.case,"). ", " Hoteling's T2 confidence interval ", params$q, ".", sep="")

loadings.cap<-paste("Ranked loadings plot. (", 1-params$q, ", ", params$q, ") quartiles represented by dashed red lines.", sep="")

loadings.cap2<-paste("Ranked loadings plot. Top loadings are labelled. (", 1-params$q, ", ", params$q, ") quartiles.", sep="")

loadings.cap3<-paste("PCA loadings plot. (", 1-params$q, ", ", params$q, ") quartiles labelled.", sep="")

boxplots.cap<-paste("Bar plots top loadingds from PC1 only. ", control, " = ", colours[1], ". ", case, " = ", colours[2], ". Student's t-test p-values <0.05, 0.01, 0.001 are represented by *, **, and ***, respectively. Mean +/- SEM." , sep="")

boxplots.cap2<-paste("Bar plots top loadingds from PC2 only. ", control, " = ", colours[1], ". ", case, " = ", colours[2], ". Student's t-test p-values <0.05, 0.01, 0.001 are represented by *, **, and ***, respectively. Mean +/- SEM." , sep="")

colours<-c(params$col_control,params$col_case)


#order the Class levels so the control is first
Data$Class<-factor(Data$Class, levels=c(control, case))

variables<-colnames(Data)

components<-3
n.control<-dim(Data[Data$Class==control,])[1]
n.case<-dim(Data[Data$Class==case,])[1]

# set the random number seed for reproducibility - shouldnt be necessary with 10 fold cross validation
set.seed(34)

# scale data
size<-dim(Data)
X<-Data
X_allscaled<-X


## ----pca_analysis, include=FALSE---------------------------------------------
pca.alldata<-opls(Data[,3:dim(Data)[2]],scaleC=params$scale)
scores.pca<-getScoreMN(pca.alldata)
variance<-getPcaVarVn(pca.alldata)
pca.loadings<-as.data.frame(getLoadingMN(pca.alldata))
colnames(pca.loadings)<-c(paste("PC1 (",round(variance[1],1),"%)",sep=""),paste("PC1 (",round(variance[2],1),"%)",sep=""))

## ----pca_loadings, include=FALSE---------------------------------------------

pc1.loadings<-rbind(subset(pca.loadings,pca.loadings[,1]>quantile(pca.loadings[,1],params$q)), subset(pca.loadings,pca.loadings[,1]<quantile(pca.loadings[,1],1-params$q)))
pc2.loadings<-rbind(subset(pca.loadings,pca.loadings[,2]>quantile(pca.loadings[,2],params$q)), subset(pca.loadings,pca.loadings[,2]<quantile(pca.loadings[,2],1-params$q)))
sig.loadings<-rbind(pc1.loadings, pc2.loadings)
pca.loadings$labels<-ifelse(rownames(pca.loadings) %in% rownames(sig.loadings), rownames(pca.loadings), "")


p.loadings<-ggplot(pca.loadings, aes(x = pca.loadings[,1], y = pca.loadings[,2]))+geom_point()+
geom_text_repel(aes(label = labels), size = 3, max.overlaps=Inf)+
xlim(c(min(pca.loadings[,1])*1.2,  max(pca.loadings[,1])*1.2))+
ylim(c(min(pca.loadings[,2])*1.2,  max(pca.loadings[,2])*1.2))+
xlab(paste("PC1 (",round(variance[1],1),"%)",sep=""))+
ylab(paste("PC1 (",round(variance[2],1),"%)",sep=""))



## ----pca_ranked_loadings, include=FALSE--------------------------------------


pca.loadings$PC1.labels<-ifelse(rownames(pca.loadings) %in% rownames(pc1.loadings), rownames(pca.loadings), "")
rank.loadings<-pca.loadings[order(pca.loadings[,1]),]

p.rankedloadings<-ggplot(rank.loadings, aes(x=rank(rank.loadings[,1]), y=rank.loadings[,1]))+geom_point()+
xlab("Rank")+
ylab("PC1 Loadings")+
geom_hline(yintercept=quantile(pca.loadings[,1],params$q), linetype="dashed", color ="red", size=0.5)+
geom_hline(yintercept=quantile(pca.loadings[,1],1-params$q), linetype="dashed", color ="red", size=0.5)

p.rankedloadings2<-ggplot(rank.loadings, aes(x=rank(rank.loadings[,1]), y=rank.loadings[,1]))+geom_point()+
geom_text_repel(aes(label = PC1.labels), size = 3, max.overlaps  = Inf)+
xlab("Rank")+
ylab("PC1 Loadings")


data.top<-cbind(Data[,1:2],Data[,colnames(Data) %in% rownames(pc1.loadings)])
data.top2<-cbind(Data[,1:2],Data[,colnames(Data) %in% rownames(pc2.loadings)])

variables.top<-colnames(data.top)

Class<-levels(data.top$Class)
mean<-rep(0,length(levels(data.top$Class)))
sd<-rep(0,length(levels(data.top$Class)))
sem<-rep(0,length(levels(data.top$Class)))

plot_list<-list()

for(k in 3:dim(data.top)[2]){

stats<-data.frame(Class,mean,sd,sem)

for(i in 1:length(levels(data.top$Class))){
stats[i,2]<-mean(data.top[data.top$Class==levels(data.top$Class)[i],k])
stats[i,3]<-sd(data.top[data.top$Class==levels(data.top$Class)[i],k])
stats[i,4]<-sd(data.top[data.top$Class==levels(data.top$Class)[i],k])/sqrt(length(data.top[data.top$Class==levels(data.top$Class)[i],k]))
}

if (t.test(data.top[,k]~data.top$Class)$p.value==0){
sig<-"***"
} else if (t.test(data.top[,k]~data.top$Class)$p.value<0.001){
sig<-"***"
} else if (t.test(data.top[,k]~data.top$Class)$p.value<0.01){
sig<-"**"
} else if (t.test(data.top[,k]~data.top$Class)$p.value<0.05){
sig<-"*"
} else
sig<-"NS"

my_plot<- ggplot(stats, aes(x=Class, y=mean, fill=Class)) + geom_bar(stat="identity", position=position_dodge(), color="black")+ 
geom_errorbar(aes(ymin=mean-sem, ymax=mean+sem), width=.2,position=position_dodge(.9))+
labs(title=variables.top[k], x="",y="Spectral Integral (AU)")+
scale_fill_manual(values=alpha(colours, 0.5))+
scale_y_continuous(expand = c(0, 0), limits = c(0,max(stats[,2])*1.3))+
theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
panel.background = element_blank(),axis.line = element_line(colour = "black"), axis.text.x=element_blank(), legend.position="none")+
geom_signif(comparisons = list(levels(data.top$Class)),annotations=sig, y_position=max(stats[,2])*1.2)

plot_list<-c(plot_list, list(my_plot))
}


## ----plots_PC2, include=FALSE------------------------------------------------
data.top2<-cbind(Data[,1:2],Data[,colnames(Data) %in% rownames(pc2.loadings)])

variables.top2<-colnames(data.top2)

Class<-levels(data.top2$Class)
mean2<-rep(0,length(levels(data.top2$Class)))
sd2<-rep(0,length(levels(data.top$Class)))
sem2<-rep(0,length(levels(data.top2$Class)))

plot_list2<-list()

for(k in 3:dim(data.top2)[2]){

stats2<-data.frame(Class,mean2,sd2,sem2)

for(i in 1:length(levels(data.top2$Class))){
stats2[i,2]<-mean(data.top2[data.top2$Class==levels(data.top2$Class)[i],k])
stats2[i,3]<-sd(data.top2[data.top2$Class==levels(data.top2$Class)[i],k])
stats2[i,4]<-sd(data.top2[data.top2$Class==levels(data.top2$Class)[i],k])/sqrt(length(data.top2[data.top2$Class==levels(data.top2$Class)[i],k]))
}

if (t.test(data.top2[,k]~data.top2$Class)$p.value<0.001){
sig2<-"***"
} else if (t.test(data.top2[,k]~data.top2$Class)$p.value<0.01){
sig2<-"**"
} else if (t.test(data.top2[,k]~data.top2$Class)$p.value<0.05){
sig2<-"*"
} else
sig2<-"NS"

my_plot2<- ggplot(stats2, aes(x=Class, y=mean2, fill=Class)) + geom_bar(stat="identity", position=position_dodge(), color="black")+ 
geom_errorbar(aes(ymin=mean2-sem2, ymax=mean2+sem2), width=.2,position=position_dodge(.9))+
labs(title=variables.top2[k], x="",y="Spectral Integral (AU)")+
scale_fill_manual(values=alpha(colours, 0.5))+
scale_y_continuous(expand = c(0, 0), limits = c(0,max(stats2[,2])*1.3))+
theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
panel.background = element_blank(),axis.line = element_line(colour = "black"), axis.text.x=element_blank(), legend.position="none")+
geom_signif(comparisons = list(levels(data.top2$Class)),annotations=sig2, y_position=max(stats2[,2])*1.2)

plot_list2<-c(plot_list2, list(my_plot2))
}


## ----ttest, include=FALSE----------------------------------------------------
variables<-colnames(Data)
ttest.summary<-matrix(0,length(variables)-2,1)
rownames(ttest.summary)<-variables[3:dim(Data)[2]]
sig.level<-matrix(0,length(variables)-2,1)
for (i in 3:length(variables)){
ttest.summary[i-2,1]<-as.numeric(t.test(Data[,i]~Data$Class)$p.value,5)
}
ttest.summary<-ttest.summary[order(ttest.summary),]
for (i in 3:length(variables)){
if (ttest.summary[i-2]<0.001){
sig.level[i-2,1]<-"***"
} else if (ttest.summary[i-2]<0.01){
sig.level[i-2,1]<-"**"
} else if (ttest.summary[i-2]<0.05){
sig.level[i-2,1]<-"*"
} else
sig.level[i-2,1]<-"NS"
}

bonf<-ttest.summary*(dim(Data)[2]-2)


ttest.summary<-cbind(round(ttest.summary,5),sig.level,round(bonf,5))
colnames(ttest.summary)<-c("p-value","significance","Bonferronni")

#ttest of top loadings only
sig.loadings<-rbind(pc1.loadings, pc2.loadings)
ttest.top<-ttest.summary[rownames(ttest.summary) %in% rownames(sig.loadings),]


## ----------------------------------------------------------------------------
pca.alldata


## ----PCA_summary_plot, dpi=600, fig.cap="PCA Summary", echo=FALSE------------
plot(pca.alldata)


## ----PCA plots, dpi=600, fig.cap=pca.plot.cap, echo=FALSE--------------------
pairs(scores.pca, pch=c(21,22)[Data$Class],bg=Data$Class, cex=1.5)


## ----PC1vPC2_plot, dpi=600, fig.cap=pca.hotel.cap, echo=FALSE----------------
plot(scores.pca[,1:2], pch=c(21,22)[Data$Class],bg=Data$Class, cex=1.5,xlab=paste("PC1 (",round(variance[1],1),"%)",sep=""), ylab=paste("PC2 (",round(variance[2],1),"%)",sep=""), xlim=c( min(scores.pca[,1])*1.8,  max(scores.pca[,1])*1.8), ylim=c( min(scores.pca[,2])*1.8,  max(scores.pca[,2])*1.8))
text(scores.pca[,1], scores.pca[,2], labels=Data$ID, cex= 0.7, pos=3)
ordiellipse(scores.pca[,1:2],Data$Class, conf=params$q, col=colours[1], show.groups=control)
ordiellipse(scores.pca[,1:2],Data$Class, conf=params$q, col=colours[2], show.groups=case)


## ----ranked_loadings, dpi=600, fig.cap=loadings.cap, echo=FALSE--------------
print(p.rankedloadings)


## ----ranked_loadings_labels, dpi=600, fig.cap=loadings.cap2, echo=FALSE------
print(p.rankedloadings2)


## ----loadings_plot, dpi=600, fig.cap=loadings.cap3, echo=FALSE---------------
print(p.loadings)


## ----ttest_table, echo=FALSE-------------------------------------------------
knitr::kable(ttest.top, caption = "Top loadings. Student's t-test p-values <0.05, 0.01, 0.001 are represented by *, **, and ***, respectively. NS; not significant.")


## ----ttest_table_all, echo=FALSE---------------------------------------------
knitr::kable(ttest.summary, caption = "All variables. Student's t-test p-values <0.05, 0.01, 0.001 are represented by *, **, and ***, respectively. NS; not significant.")


## ----boxplots, echo=FALSE, dpi=600, fig.cap=boxplots.cap---------------------


ce<-ceiling(length(plot_list)/6)

if (ce==1){
  grid.arrange(grobs=plot_list, ncol=3, nrow=3)
} else {
for (l in 1:ce){
low<-6*l-5

if (l<ce) {
high<-6*l
} else {
high<-length(plot_list)
}
grid.arrange(grobs=plot_list[low:high], ncol=3, nrow=3)

}
}


## ----boxplots_pc2loadings, echo=FALSE, dpi=600, fig.cap=boxplots.cap2--------


ce<-ceiling(length(plot_list2)/6)

if (ce==1){
  grid.arrange(grobs=plot_list2, ncol=3, nrow=3)
} else {
for (l in 1:ce){
low<-6*l-5

if (l<ce) {
high<-6*l
} else {
high<-length(plot_list2)
}
grid.arrange(grobs=plot_list2[low:high], ncol=3, nrow=3)

}
}


## ----save_workspace----------------------------------------------------------

save.image(file=paste(params$filename,"_",format(Sys.time(), "%e%h%g"),"_",format(Sys.time(), "%H%M"), ".RData", sep=""))


