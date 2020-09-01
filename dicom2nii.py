import os
import os.path as osp
import dicom2nifti
import shutil

#dicom2nifti.convert_dir.convert_directory('T1WI', 'test')
#dicom2nifti.dicom_series_to_nifti('DWI', './test/test.nii', reorient_nifti=True)

def process_one(f_in, f_out):
    print (f'processing {f_in} ...')
    dicom2nifti.convert_dir.convert_directory(f_in, f_out)

def main(f_path):

    f_path = f_path.rstrip('/')
    f_out = f_path+'_nii'
    os.makedirs(f_out, exist_ok=True)
    fs = os.listdir(f_path)
    for f in fs:
        print (f'processing {f} ...')
        f_p = osp.join(f_path, f)
        gt_p = osp.join(f_p, f+'.nii')
        if not osp.exists(gt_p):
            print (f'{f}.nii not exist !!!')
            continue

        f_out_p = osp.join(f_out, f)
        if osp.exists(f_out_p):
            print (f'{f_out_p} exists !!!')
            continue

        os.makedirs(f_out_p, exist_ok=True)
        dwi1_p = osp.join(f_p, 'DWI1')
        os.makedirs(dwi1_p, exist_ok=True)
        dwi2_p = osp.join(f_p, 'DWI2')
        os.makedirs(dwi2_p, exist_ok=True) 
        dwi_p = osp.join(f_p, 'DWI')
        dwi_list = os.listdir(dwi_p)
        #dwi_list.sort()
        dwi_list.sort(key = lambda x:int(x.split('.')[0].split('-')[-1]))

        if len(dwi_list) %2 != 0:
            print ('dwi len is not even !!!')
            os.removedirs(f_out_p)
            raise

        dwi1_list = dwi_list[:len(dwi_list)//2]
        dwi2_list = dwi_list[len(dwi_list)//2:]

        for i in dwi1_list:
            i_p = osp.join(dwi_p, i)
            shutil.copy(i_p, dwi1_p)
        for i in dwi2_list:
            i_p = osp.join(dwi_p, i)
            shutil.copy(i_p, dwi2_p)


        t1c_p = osp.join(f_p, 'T1+C')
        t1wi_p = osp.join(f_p, 'T1WI')
        t2wi_p = osp.join(f_p, 'T2WI')

        process_one(dwi2_p, f_out_p)    
        process_one(t1c_p, f_out_p)
        process_one(t1wi_p, f_out_p)
        process_one(t2wi_p, f_out_p)

        shutil.copy(gt_p, f_out_p)

if __name__=='__main__':
    import fire
    fire.Fire(main)

