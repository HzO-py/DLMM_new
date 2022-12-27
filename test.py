def weights_fusion(w1,w2):
    w=[]
    for i in range(len(w1)):
        w.append(w1[i]*w2[i])
    weights=[x/sum(w) for x in w]
    return weights
x=weights_fusion([0.1,0.6,0.3],[0.05,0.9,0.05])


                # if os.path.exists(voicepath):
                #     flg=True
                #     biopath=os.path.join(*splits[:-2])
                #     if os.path.exists(biopath):
                #         for bioname in os.listdir(biopath):
                #             if bioname.endswith('.csv'):
                #                 bioname=bioname.split('.')[0]
                #                 if sample_person_id==int(bioname.split('-')[0]) and sample_sample_id==int(bioname.split('-')[-1]):
                #                     biopath=os.path.join(biopath,bioname+'.csv')
                #                     samples.append([root_path_3,voicepath,biopath,label])
                #                     if ageclass!=-1:
                #                         score_distribute_bio[ageclass][int(label//0.1)]+=1
                #                     flg=False
                #                     break
                #     if flg:
                #         samples.append([root_path_3,voicepath,label])