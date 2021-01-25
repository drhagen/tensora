import platform
import subprocess
from pathlib import Path

from distutils.command.build import build
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel

project_dir = Path(__file__).parent.resolve()
taco_source_dir = project_dir.joinpath('src/taco')
taco_build_dir = project_dir.joinpath('build/taco/')
taco_install_dir = project_dir.joinpath('src/tensora/taco/')


class TensoraBuild(build):
    def run(self):
        # Build taco
        OS = platform.system()
        if OS == "Linux":
            install_path = r'-DCMAKE_INSTALL_RPATH=\$ORIGIN/../lib'
        elif OS == "Darwin":
            install_path = r'-DCMAKE_INSTALL_RPATH=@loader_path/../lib'
        else:
            raise NotImplementedError(f'Tensora cannot be installed on {OS}')

        taco_build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(['cmake', str(taco_source_dir),
                               '-DCMAKE_BUILD_TYPE=Release',
                               f'-DCMAKE_INSTALL_PREFIX={taco_install_dir}',
                               install_path],
                              cwd=taco_build_dir)
        subprocess.check_call(['make', '-j8'], cwd=taco_build_dir)
        subprocess.check_call(['make', 'install'], cwd=taco_build_dir)

        super().run()


class TensoraBdistWheel(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)

        # Ensure that platform tag is included because binaries are platform-specific
        self.root_is_pure = False


setup(
    name='tensora',
    version='0.0.4',

    description='Library for dense and sparse tensors built on the tensor algebra compiler.',
    long_description=Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    keywords='tensor sparse matrix array',

    author='David Hagen',
    author_email='david@drhagen.com',
    url='https://github.com/drhagen/tensora',
    license='MIT',

    package_dir={'': 'src'},
    packages=find_packages('src'),
    package_data={'tensora': ['taco/bin/taco', 'taco/lib/libtaco.*']},

    install_requires=Path('requirements.txt').read_text(encoding='utf-8').splitlines(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    cmdclass={
        'build': TensoraBuild,
        'bdist_wheel': TensoraBdistWheel,
    },
    zip_safe=False,
)
