import os
import cv2
import h5py
import numpy as np
import dask.array as da
from unittest import TestCase
from dask.distributed import Client
from moseq2_pca.util import recursive_find_h5s, read_yaml
from moseq2_pca.helpers.parameters import (
    MouseProcessingParams,
    SVDConfig,
    DaskConfig,
    ChangepointParams,
    MaskParams,
    create_dataclass_from_dict,
)
from moseq2_pca.pca.util import (
    mask_data,
    train_pca_dask,
    apply_pca_dask,
    get_changepoints_dask,
)


class TestPCAUtils(TestCase):

    def test_mask_data(self):
        nframes = 10

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        tmp_image = np.ones((80, 80), dtype='int8')
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
        center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

        frames = np.tile(tmp_image, (nframes, 1, 1))
        init_frames = frames.copy()
        mask = frames.reshape(-1, frames.shape[1] * frames.shape[2])

        new_data = np.zeros(frames.shape)

        assert(mask.shape == (nframes, 6400))
        assert(new_data.shape == (nframes, 80, 80))

        test_out = mask_data(frames, mask, new_data)
        frames[mask] = new_data[mask]

        assert (frames.any() > 0) == (init_frames.all() == 0)
        assert frames.all() == test_out.all()

    def test_train_pca_dask(self):

        input_dir = 'data/proc/'
        config_file = 'data/config.yaml'

        config_data = read_yaml(config_file)

        h5s, dicts, yamls = recursive_find_h5s(input_dir)

        h5ps = [h5py.File(h5, mode='r') for h5 in h5s]
        arrays = [da.from_array(fp['/frames'], chunks=(1000, -1, -1)) for fp in h5ps]
        stacked_array = da.concatenate(arrays, axis=0)

        stacked_array = da.where(
            da.logical_or(stacked_array < 10, stacked_array > 100), 0, stacked_array
        )

        mouse_proc_params, _ = create_dataclass_from_dict(MouseProcessingParams, config_data)
        svd_config, _ = create_dataclass_from_dict(SVDConfig, config_data)
        dask_config, _ = create_dataclass_from_dict(DaskConfig, config_data)
        svd_config.chunk_size = 1000

        client = Client(processes=True)

        output_dict = train_pca_dask(
            dask_array=stacked_array,
            mask=None,
            mouse_proc_params=mouse_proc_params,
            svd_config=svd_config,
            dask_config=dask_config,
            client=client,
        )
        client.restart()
        client.close()
        for fp in h5ps:
            assert fp['frames'].shape == (900, 80, 80)
            fp.close()

        assert 'components' in output_dict.keys()
        assert 'singular_values' in output_dict.keys()
        assert 'explained_variance' in output_dict.keys()
        assert 'explained_variance_ratio' in output_dict.keys()
        assert 'mean' in output_dict.keys()

        # check that all the h5 files are closed by ensuring an exception is raised
        for fp in h5ps:
            try:
                print(fp['frames'].keys()) # line that's meant to raise a ValueError
                assert False, 'h5 File pointer is still open' # added assertion here to fail test in case file is still open
            except ValueError as e:
                assert isinstance(e, ValueError)

    def test_apply_pca_dask(self):

        input_dir = 'data/proc/'
        pca_path = 'data/_pca/pca.h5'
        save_file = 'data/_pca/dask_test_pca_scores.h5'
        config_file = 'data/config.yaml'

        config_data = read_yaml(config_file)

        with h5py.File(pca_path, 'r') as f:
            pca_components = f['components'][()]

        mouse_proc_params, _ = create_dataclass_from_dict(MouseProcessingParams, config_data)
        svd_config, _ = create_dataclass_from_dict(SVDConfig, config_data)
        mask_params = MaskParams()

        h5s, dicts, yamls = recursive_find_h5s(input_dir)

        svd_config.chunk_size = 100
        client = Client(processes=True)

        apply_pca_dask(
            pca_components=pca_components,
            h5s=h5s,
            yamls=yamls,
            mouse_proc_params=mouse_proc_params,
            save_file=save_file,
            mask_params=mask_params,
            svd_config=svd_config,
            client=client,
        )

        client.restart()

        assert os.path.exists(save_file)
        os.remove(save_file)

        svd_config.missing_data = True

        apply_pca_dask(
            pca_components=pca_components,
            h5s=h5s,
            yamls=yamls,
            mouse_proc_params=mouse_proc_params,
            save_file=save_file,
            mask_params=mask_params,
            svd_config=svd_config,
            client=client,
        )

        client.restart()
        client.close()

        assert os.path.exists(save_file)
        os.remove(save_file)

        # testing list comprehension file closing
        h5_file_pointers = [h5py.File(h5, 'r') for h5 in h5s]

        [h5p.close() for h5p in h5_file_pointers]

        try:
            for h5p in h5_file_pointers:
                print(h5p.keys())
                assert False, 'h5 File pointer is still open'
        except TypeError as e:
            assert isinstance(e, TypeError)

    def test_get_changepoints_dask(self):

        input_dir = 'data/proc/'
        pca_path = 'data/_pca/pca.h5'
        save_file = 'data/_pca/test_changepoints.h5'
        config_file = 'data/config.yaml'

        config_data = read_yaml(config_file)

        with h5py.File(pca_path, 'r') as f:
            pca_components = f['components'][()]

        changepoint_params, _ = create_dataclass_from_dict(ChangepointParams, config_data)
        mask_params = MaskParams()

        missing_data = False

        h5s, dicts, yamls = recursive_find_h5s(input_dir)

        chunk_size = 100
        client = Client(processes=True)

        get_changepoints_dask(
            changepoint_params=changepoint_params,
            pca_components=pca_components,
            h5s=h5s,
            yamls=yamls,
            save_file=save_file,
            chunk_size=chunk_size,
            mask_params=mask_params,
            missing_data=missing_data,
            client=client,
        )
        client.restart()
        client.close()
        client = Client(processes=True)

        assert os.path.exists(save_file)
        os.remove(save_file)

        missing_data_save_file = 'data/_pca/dask_test_pca_scores.h5'

        missing_data = True

        svd_config, _ = create_dataclass_from_dict(SVDConfig, config_data)
        svd_config.chunk_size = chunk_size
        mouse_proc_params, _ = create_dataclass_from_dict(MouseProcessingParams, config_data)

        apply_pca_dask(
            pca_components=pca_components,
            h5s=h5s,
            yamls=yamls,
            mouse_proc_params=mouse_proc_params,
            save_file=missing_data_save_file,
            mask_params=mask_params,
            svd_config=svd_config,
            client=client,
        )

        assert os.path.exists(missing_data_save_file)

        get_changepoints_dask(
            changepoint_params=changepoint_params,
            pca_components=pca_components,
            h5s=h5s,
            yamls=yamls,
            save_file=save_file,
            chunk_size=chunk_size,
            mask_params=mask_params,
            missing_data=missing_data,
            client=client,
            fps=30,
            pca_scores=missing_data_save_file,
        )
        client.restart()
        client.close()

        assert os.path.exists(save_file)
        assert os.path.exists(missing_data_save_file)
        os.remove(save_file)
        os.remove(missing_data_save_file)
