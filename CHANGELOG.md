# CHANGELOG



## v0.5.0 (2023-11-14)

### Feature

* feat: Add KNN Classifier (#20) ([`bb77222`](https://github.com/cmnemoi/cmnemoi-learn/commit/bb7722247250b808678f2fdc6cc55a8f29b4589a))

### Style

* style: update black rules to allow lines up to 100 characters (#19) ([`93e438b`](https://github.com/cmnemoi/cmnemoi-learn/commit/93e438bbba24d149f5f42cdca94f79eb28e3ebe8))


## v0.4.0 (2023-09-16)

### Feature

* feat: New public API making usage more simple (#17) ([`db1aa7e`](https://github.com/cmnemoi/cmnemoi-learn/commit/db1aa7e502d7168b1a149c1d82c9681eff563769))


## v0.3.0 (2023-09-15)

### Ci

* ci: fix dependencies installation in pipelines (#16) ([`481f300`](https://github.com/cmnemoi/cmnemoi-learn/commit/481f300391f9134125a18eb37973de4a7cc4dfe9))

### Feature

* feat: Add `LogisticRegression` (#14) ([`becd1fc`](https://github.com/cmnemoi/cmnemoi-learn/commit/becd1fcabc948badba8a0930356085a2d031b6c4))


## v0.2.2 (2023-08-30)

### Chore

* chore: bump version to v0.2.1 (#12) ([`3ee2288`](https://github.com/cmnemoi/cmnemoi-learn/commit/3ee22881f5964d7dbf5d4c2579f801d1f48a5b11))

### Ci

* ci: allow to run `Publish package to PyPI` pipeline manually (#13) ([`661ad40`](https://github.com/cmnemoi/cmnemoi-learn/commit/661ad40514a826d963f2aae8f5b1b44cc30f5baa))

### Fix

* fix: reduce number of dependencies (#15) ([`3125b88`](https://github.com/cmnemoi/cmnemoi-learn/commit/3125b885576265685f2c0e2b534b8584c84e1c3d))


## v0.2.1 (2023-08-06)

### Ci

* ci: Only check if code has to be formatted in lint step instead of formatting it (#9) ([`23a5160`](https://github.com/cmnemoi/cmnemoi-learn/commit/23a51609f3ef26f17885531f246c06c1c8ec6f8b))

* ci: fix CD comitter name and email (#8) ([`04ad2af`](https://github.com/cmnemoi/cmnemoi-learn/commit/04ad2af983cffea8598321bcd98a9c096cd596c7))

### Documentation

* docs: Add repository link to package metadata (#7) ([`173f743`](https://github.com/cmnemoi/cmnemoi-learn/commit/173f743809765cae0a66a0a1ec7e1844a78759cc))

* docs: Add code quality badges (#6) ([`e87f13f`](https://github.com/cmnemoi/cmnemoi-learn/commit/e87f13f9a74e0a46345218d35b03240b7c7ef461))

### Fix

* fix: `LinearRegression` predictions are now correct in case of under determined systems (n &lt; m) (#11) ([`957328b`](https://github.com/cmnemoi/cmnemoi-learn/commit/957328b2ba857f51fc6b0e0d7d917331e26873bc))

### Refactor

* refactor: Add abstract model classes for reusability (#10) ([`30d520b`](https://github.com/cmnemoi/cmnemoi-learn/commit/30d520b485f8a47d7bbd4a253d1e8643e9f9947d))


## v0.2.0 (2023-08-06)

### Chore

* chore: initial commit ([`11d3c19`](https://github.com/cmnemoi/cmnemoi-learn/commit/11d3c19600326281ef68a4121ca19e021e6f67b3))

### Ci

* ci: fix the way we give a token to CD pipeline (#5) ([`19edfc6`](https://github.com/cmnemoi/cmnemoi-learn/commit/19edfc6ad02bfe29592e94d28520b78d37025657))

* ci: trying to fix Github Release  (#3)

* ci: publish new pypi package at each new tag created

* ci: trying to fix token

* ci: test

* remove test ([`1b0b237`](https://github.com/cmnemoi/cmnemoi-learn/commit/1b0b237c059a10f0b1b63d3353f0343fe39fa23d))

* ci: publish new pypi package at each new tag created (#2) ([`0293b03`](https://github.com/cmnemoi/cmnemoi-learn/commit/0293b03823de2df8211e666494db9ba478bb50d3))

* ci: Add CI/CD pipeline (#1)

* chore: add a .gitignore

* chore: add a Makefile

* docs: update README

* feat: add first code snippets

* ci: add CI pipeline

* ci: trying to cache python dependencies

* ci: remove install depedencies step

* style: apply linter fixes

* ci: add CD pipeline

* ci: fix CD pipeline trigger

* ci: add GitHub Release creation to CD pipeline

* docs: update README

* ci: only trigger CI on Python files ([`1f70d63`](https://github.com/cmnemoi/cmnemoi-learn/commit/1f70d6339b0adfe8c10d529dfce2341c8eaf6db5))

### Feature

* feat: Add Linear Regression (#4)

* style: fix linter

* test: add tests

* feat: add LinearRegression .fit and .predict method

* chore: bump version to 0.2

* refactor: put fixtures parameters into constants

* fix: add a bias column to the LinearRegression model

* style: apply linter fixes ([`ac0366c`](https://github.com/cmnemoi/cmnemoi-learn/commit/ac0366c456d07325a95c6334b2fc6380b82e669b))
