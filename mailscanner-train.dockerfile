FROM mailscanner-base

#mount in the local var 
VOLUME /mailscanner

#make the model
WORKDIR /mailscanner
CMD make var/data/replies.weights