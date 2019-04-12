<div align="center">

# Sitio Web de la Conferencia SciPyLatam 2019

[![Build Status][build-badge]][build]
[![MIT License][license-badge]][LICENSE]
[![Python Status](https://img.shields.io/badge/Python-%3E%3D3.5-blue.svg?longCache=true&style=flat-square)](https://www.python.org/)
[![PRs Welcome][prs-badge]][prs] 
[![GitHub issues](https://img.shields.io/github/issues/scipy-latinamerica/scipylatam-webpage-2019.svg?style=flat-square)](scipy-latinamerica/scipylatam-webpage-2019/issues)
[![Twitter SciPyLatam](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/scipyla)

Este es el repositorio del sitio web de la comunidad servido por [Github](scipy-latinamerica/scipylatam-webpage-2019), y
construido utilizando [lektor](https://www.getlektor.com).

</div>

# 🔀 Flujo de trabajo

Hay 2 ramas de trabajo en git, `develop` y `production`.

## ⤴️ Develop

Es la rama por defecto y se despliega a través de gh-pages con Lektor a
https://github.com/scipy-latinamerica/scipylatam-webpage-2019-develop
(en https://scipy-latinamerica.github.io/scipylatam-webpage-2019-develop/)

## ⤴️ Production

Después de que se han ejecutado las pruebas de calidad (QA), los cambios
realizados en la rama `develop` se unen con la rama `production` y son
desplegados a través de gh-pages con Lektor a la rama `gh-pages` de
https://scipy-latinamerica.github.io/scipylatam-webpage-2019/
(pronto en http://scipy.org/conf/2019 o http://conf.scipy.org)

# 🛠 Desarrollo local

## ✅ (Opcional) Entorno Virtual

### Instalación

* Instalar [virtualenv](https://virtualenv.pypa.io/en/stable/installation/)

* ⚠️️ Requiere previa instalación de Python ⚠️

### Crear entorno virtual

```
$ virtualenv <nombre_entorno>
```

### Activar entorno virtual

#### Windows

```
$ <nombre_entorno>\Scripts\activate
```

#### Linux

```
$ source <nombre_entorno>/bin/activate
```

## ✅ Instala Lektor

### Usando pip:
```
$ pip install -U Lektor fabric
```

### Usando conda:

```
$ conda install lektor fabric -c conda-forge
```

## ✅ Instala (reinstala) los `plugins` locales

```
$ lektor plugins reinstall
```

## ✅ Instala nvm

Necesitamos instalar [node](https://github.com/creationix/nvm)

Usando cURL:

```
$ curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.34.0/install.sh | bash
```

Después instalamos Node:

```
$ nvm install --lts
```

Instalar paquetes necesarios para el bundle:
```
$ cd webpack
$ npm i
```

## ✅ Corre el servidor local

```
$ lektor server -f webpack
```

## ⁉️ Problemas comunes

* 🔴 Si en algun momento luego de instalar python3 y crear tu virtualenv. haces `lektor server` y ves este error:

```
RuntimeError: Click will abort further execution because Python 3 was configured to use ASCIas encoding for the environment.  Consult http://click.pocoo.org/python3/for mitigation steps.
```
Haz esto adentro de tu virtualenv:
```
export LC_ALL=en_us.UTF-8
export LANG=en_us.UTF-8
```

* 🔴 Si ves `jinja2.exceptions.UndefinedError: 'estimate_reading_time' is undefined` significa que necesitas instalar o reinstalar los `plugins` de lektor. Puedes hacer esto ejecutando

```
$ lektor plugins reinstall
```

# 🚀 Despliegue

Gracias a _Lektor Bot_ (Plugin de lektor conectado a Github), podemos desplegar nuestra web estática en diferentes repositorios (en la rama seleccionada en configuración - `gh-pages` en nuestro caso). Los despliegues son automáticos utilizando Travis-CI para cada PR que se
une con la rama develop (o con la rama production).

La configuración de dichos repositorios se encuentra en [scipylatam2019_webpage.lektorproject](https://github.com/scipy-latinamerica/scipylatam-webpage-2019/blob/develop/scipylatam2019_webpage.lektorproject)

## ✔️ Desarrollo

```
$ lektor deploy
```

Se desplegará el contenido de la rama `develop` en [https://scipy-latinamerica.github.io/scipylatam-webpage-2019/](https://scipy-latinamerica.github.io/scipylatam-webpage-2019/)

## ✔️✔️ Producción

```
$ lektor deploy production
```

Se desplegará el contenido de la rama `production` en [http://scipyla.org/conf/2019](http://scipyla.org/conf/2019)


[build-badge]: https://img.shields.io/travis/scipy-latinamerica/scipylatam-webpage-2019.svg?style=flat-square
[build]: https://travis-ci.org/scipy-latinamerica/scipylatam-webpage-2019
[license-badge]: https://img.shields.io/npm/l/all-contributors.svg?style=flat-square
[license]: https://github.com/scipy-latinamerica/scipylatam-webpage-2019/blob/develop/LICENSE.txt
[prs-badge]: https://img.shields.io/badge/Issues-welcome-brightgreen.svg?style=flat-square
[prs]: https://github.com/scipy-latinamerica/scipylatam-webpage-2019/issues/new
[github-watch-badge]: https://img.shields.io/github/watchers/kentcdodds/all-contributors.svg?style=social
[github-watch]: https://github.com/kentcdodds/all-contributors/watchers
