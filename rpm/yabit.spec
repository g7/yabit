Name:           yabit
Version:        0.0.1
Release:        1
Summary:        Yet Another BootImage Tool

Source:         %{name}-%{version}.tar.bz2

License:        BSD-3-Clause
URL:            https://github.com/g7/yabit

BuildArch:      noarch
BuildRequires:  python3-base

%description
yabit is a python written, device tree-aware tool to create, extract and
update Android BootImages ("boot.img").
It depends only on a reasonably up-to-date Python interpreter (3.4+)
and the standard library.

%prep
%setup -q -n %{name}-%{version}/%{name}

%build
python3 setup.py build

%install
python3 setup.py install --skip-build --root %{buildroot}

mkdir -p ${RPM_BUILD_ROOT}/%{_bindir}
ln -sf /usr/share/yabit/yabit.py %{buildroot}/%{_bindir}/yabit

%files
#%license LICENCE
%doc README.md
/usr/share/yabit
/usr/lib/python*
%{_bindir}

%changelog
* Tue Apr 03 2018 Eugenio "g7" Paolantonio <me@medesimo.eu> - 0.0.1-1
- Initial release
