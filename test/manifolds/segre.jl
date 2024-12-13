using Manifolds, Test, Random, LinearAlgebra, FiniteDifferences

@testset "Segre Manifold" begin
    # Manifolds to test
    Ms = [
        Segre(10),
        Segre(7, 2),
        Segre(7, 9, 9),
        Segre(9, 3, 6, 6),
        MetricManifold(Segre(10), WarpedMetric(1.2025837056880606)),
        MetricManifold(Segre(2, 9), WarpedMetric(1.1302422072971439)),
        MetricManifold(Segre(9, 6, 10), WarpedMetric(1.4545138169484464)),
        MetricManifold(Segre(9, 3, 8, 10), WarpedMetric(1.396673190458706)),
    ]

    # Vs[i] is the valence of Ms[i]
    Vs = [(10,), (7, 2), (7, 9, 9), (9, 3, 6, 6), (10,), (2, 9), (9, 6, 10), (9, 3, 8, 10)]

    # n ≥ k, for same n,k X is in TpM and can be scaled by l
    unit_p(n, k) = 1 / sqrt(k) .* [ones(k)..., zeros(n - k)...]
    unit_X(n, k; l=1.0) = l / sqrt(n - k) .* [zeros(k)..., ones(n - k)...]

    # ps[i] is a point on Ms[i]
    ps = [
        [[0.5], unit_p(10, 4)],
        [[0.6], unit_p(7, 3), unit_p(2, 1)],
        [[0.7], unit_p(7, 3), unit_p(9, 5), unit_p(9, 4)],
        [[0.8], unit_p(9, 3), unit_p(3, 1), unit_p(6, 4), unit_p(6, 3)],
        [[0.9], unit_p(10, 4)],
        [[1.0], unit_p(2, 1), unit_p(9, 5)],
        [[1.1], unit_p(9, 3), unit_p(6, 5), unit_p(10, 4)],
        [[1.2], unit_p(9, 3), unit_p(3, 1), unit_p(8, 4), unit_p(10, 4)],
    ]

    # qs[i] is a point on Ms[i] that is connected to ps[i] by a geodesic and uses the closest representative to ps[i]
    # (tricked by only switching only one entry to zero)
    qs = [
        [[0.1], unit_p(10, 3)],
        [[0.2], unit_p(7, 2), unit_p(2, 2)],
        [[0.3], unit_p(7, 2), unit_p(9, 4), unit_p(9, 3)],
        [[0.4], unit_p(9, 3), unit_p(3, 1), unit_p(6, 3), unit_p(6, 2)],
        [[0.5], unit_p(10, 3)],
        [[0.6], unit_p(2, 2), unit_p(9, 5)],
        [[0.7], unit_p(9, 2), unit_p(6, 4), unit_p(10, 3)],
        [[0.8], unit_p(9, 2), unit_p(3, 2), unit_p(8, 3), unit_p(10, 3)],
    ]

    # us[i] is a tangent vector to Ms[i] at ps[i]
    us = [
        [[0.5], unit_X(10, 4)],
        [[0.6], unit_X(7, 3), unit_X(2, 1)],
        [[0.7], unit_X(7, 3), unit_X(9, 5), unit_X(9, 4)],
        [[0.8], unit_X(9, 3), unit_X(3, 1), unit_X(6, 4), unit_X(6, 3)],
        [[0.9], unit_X(10, 4)],
        [[1.0], unit_X(2, 1), unit_X(9, 5)],
        [[1.1], unit_X(9, 3), unit_X(6, 5), unit_X(10, 4)],
        [[1.2], unit_X(9, 3), unit_X(3, 1), unit_X(8, 4), unit_X(10, 4)],
    ]

    # vs[i] is a tangent vector to Ms[i] at ps[i] such that exp(Ms[i], ps[i], t * vs[i]) is the closes representative to ps[i] for t in [-1, 1]
    vs = [
        [[0.5], unit_X(10, 5)],
        [[0.6], unit_X(7, 4), unit_X(2, 1)],
        [[0.7], unit_X(7, 4), unit_X(9, 6), unit_X(9, 5)],
        [[0.8], unit_X(9, 4), unit_X(3, 1), unit_X(6, 5), unit_X(6, 4)],
        [[0.9], unit_X(10, 5)],
        [[1.0], unit_X(2, 1), unit_X(9, 6)],
        [[1.1], unit_X(9, 4), unit_X(6, 5), unit_X(10, 5)],
        [[1.2], unit_X(9, 4), unit_X(3, 1), unit_X(8, 5), unit_X(10, 5)],
    ]

    # When testing that exp(Ms[i], ps[i], t * vs[i]) is an extremum of the length functional, we parametrize curves in Ms[i] and take a derivative along dys[i]
    dys = [
        [
            0.24768051036976024,
            0.15340874919881428,
            0.10461792180729243,
            0.1621738878738882,
            0.13788895788409475,
            0.13920478959340893,
            0.1980520199776914,
            0.225220769065586,
            0.09192923424127934,
            0.15861132814448145,
            0.07788788182581427,
            0.2033127058413462,
            0.11406283238286104,
            0.020870012361419343,
            0.2719578430552649,
            0.15170700765742606,
            0.07022129511933332,
            0.04843952826362164,
            0.2555544578976192,
            0.06681340738035137,
            0.15586613310716685,
            0.018646851863471762,
            0.1979297092698637,
            0.23753411269988997,
            0.21343310337377225,
            0.08016910818336964,
            0.1464783972931314,
            0.025950651929534232,
            0.1449831966165307,
            0.23373068876892722,
            0.11880598168141776,
            0.19121899600576953,
            0.15179084381364852,
            0.04935589775227154,
            0.1560593153225572,
            0.04411343229804097,
            0.2392666728301238,
            0.1484349911958273,
            0.17060958026859915,
            0.01433037040599356,
        ],
        [
            0.2703013619564934,
            0.1674196883015804,
            0.11417275710290055,
            0.17698535383607492,
            0.15048246250457872,
            0.15191847013635482,
            0.2161402633508973,
            0.2457903551976426,
            0.10032520193833194,
            0.1730974227854145,
            0.08500144200281991,
            0.22188141170224845,
            0.12448027862860386,
            0.022776086648557087,
            0.29679596211605436,
            0.16556252539583352,
            0.07663466003347703,
            0.05286354765112583,
            0.278894443170582,
            0.07291552728513738,
            0.17010150697308607,
            0.020349883191748984,
            0.21600678191201325,
            0.2592282859804155,
            0.23292611292832396,
            0.08749101451887183,
            0.15985638202387917,
            0.02832074493766232,
            0.1582246235190211,
            0.255077492415341,
            0.12965662340215453,
            0.20868317404203518,
        ],
        [
            0.16353440643700792,
            0.10129020125571776,
            0.06907539765598648,
            0.10707750259980159,
            0.0910430491609086,
            0.09191184484141025,
            0.13076652451317197,
            0.14870505851042468,
            0.06069752009721213,
            0.1047252743607885,
            0.05142652727905344,
            0.13423996349664308,
            0.07531152759015192,
            0.013779707893699597,
            0.17956384365300332,
            0.10016660338980309,
            0.04636455972831437,
            0.03198285359980099,
            0.16873328677427074,
            0.0441144557626597,
            0.10291272221322205,
            0.012311836110395775,
            0.1306857672142806,
            0.1568351101623881,
            0.14092209282890691,
            0.05293273783140711,
            0.09671434268855209,
            0.017134268875714943,
            0.09572711622173373,
            0.1543238480770115,
            0.07844325605769907,
            0.12625492803046978,
            0.10022195734569302,
            0.03258789894704945,
            0.10304027338340047,
            0.029126490235301297,
            0.15797905641835802,
            0.09800621027247283,
            0.11264728258206483,
            0.009461820854890907,
            0.07619899497188476,
            0.06435512726649247,
            0.14418724370624061,
            0.020482591380646117,
            0.18622326856315144,
            0.17680302406232687,
            0.11411324587011096,
            0.04559115895133764,
            0.024318253167761508,
            0.0777066933426362,
            0.16696701026147626,
            0.06641466684484068,
            0.06421006278332052,
            0.1069248857665583,
            0.12159103864374035,
            0.05073753945931583,
            0.04924168633871106,
            0.12722583436268306,
            0.18563179386637813,
            0.0687128870870491,
            0.08905886988994213,
            0.009035432164352614,
            0.049561446318667116,
            0.1347085017272271,
            0.1200359392924501,
            0.027268283817115106,
            0.04224558795450571,
            0.04315262171954351,
            0.05403346926811662,
            0.0023262859537098515,
            0.010721414504714139,
            0.13403654903491943,
            0.10987330337776338,
            0.013298908898025385,
            0.14340914694684173,
            0.17825169707180502,
            0.10868078009526573,
            0.0016445884181541684,
            0.07063958523304881,
            0.1101738445443299,
            0.11758054831650003,
            0.060827727766064016,
            0.14865227744166581,
            0.11702503201869387,
            0.09342064187053477,
            0.15738531812684184,
            0.03201521439082036,
            0.06114448967602477,
            0.10382777552410098,
            0.1431354723068435,
            0.18877393058117947,
            0.04238816261102419,
        ],
        [
            0.1715795373556408,
            0.1062732072642473,
            0.07247358541051933,
            0.11234521687243985,
            0.0955219430260548,
            0.09643347940647069,
            0.13719962830095694,
            0.15602065459839623,
            0.0636835553069129,
            0.10987726996268543,
            0.053956472834027505,
            0.14084394430027286,
            0.07901650388441198,
            0.014457605324831525,
            0.1883975482043314,
            0.10509433361797467,
            0.04864548006260954,
            0.03355626099441541,
            0.17703417838482685,
            0.046284681464683716,
            0.10797554869382073,
            0.0129175210883459,
            0.1371148981191942,
            0.164550666915154,
            0.14785480326481754,
            0.05553678192838664,
            0.1014722377736982,
            0.01797719507885178,
            0.10043644436402703,
            0.1619158624347057,
            0.08230229880238973,
            0.1324660822900412,
            0.10515241073061003,
            0.03419107175394852,
            0.10810937478733341,
            0.030559377859677404,
            0.16575088999747303,
            0.10282765922416374,
            0.11818900408120409,
            0.009927298359990544,
            0.07994763052677145,
            0.06752109970877263,
            0.15128058435355177,
            0.021490239451779292,
            0.19538458579496817,
            0.1855009091519671,
            0.1197270859333567,
            0.047834031570543896,
            0.02551459792914639,
            0.08152950063326206,
            0.17518100929637484,
            0.06968195903934253,
            0.06736889872885857,
            0.11218509200201603,
            0.1275727512737259,
            0.053233590023432004,
            0.05166414789821036,
            0.13348475269051732,
            0.19476401329869034,
            0.07209324101046329,
            0.09344015137889938,
            0.009479933332347762,
            0.051999638579474684,
            0.14133553242896432,
            0.12594114827928685,
            0.028609756342773685,
            0.04432387406709532,
            0.04527552966766203,
            0.056691664223667886,
            0.002440728384874828,
            0.011248858149159562,
            0.14063052279470464,
            0.11527855802353276,
            0.013953153258528188,
            0.15046420885860654,
            0.18702085012439856,
            0.11402736815129268,
            0.0017254945064788542,
            0.07411472372909801,
            0.11559388441532585,
            0.10893561836452229,
            0.15017707070132247,
            0.198060728501197,
            0.04447346273248423,
        ],
        [
            0.24768051036976024,
            0.15340874919881428,
            0.10461792180729243,
            0.1621738878738882,
            0.13788895788409475,
            0.13920478959340893,
            0.1980520199776914,
            0.225220769065586,
            0.09192923424127934,
            0.15861132814448145,
            0.07788788182581427,
            0.2033127058413462,
            0.11406283238286104,
            0.020870012361419343,
            0.2719578430552649,
            0.15170700765742606,
            0.07022129511933332,
            0.04843952826362164,
            0.2555544578976192,
            0.06681340738035137,
            0.15586613310716685,
            0.018646851863471762,
            0.1979297092698637,
            0.23753411269988997,
            0.21343310337377225,
            0.08016910818336964,
            0.1464783972931314,
            0.025950651929534232,
            0.1449831966165307,
            0.23373068876892722,
            0.11880598168141776,
            0.19121899600576953,
            0.15179084381364852,
            0.04935589775227154,
            0.1560593153225572,
            0.04411343229804097,
            0.2392666728301238,
            0.1484349911958273,
            0.17060958026859915,
            0.01433037040599356,
        ],
        [
            0.24768051036976024,
            0.15340874919881428,
            0.10461792180729243,
            0.1621738878738882,
            0.13788895788409475,
            0.13920478959340893,
            0.1980520199776914,
            0.225220769065586,
            0.09192923424127934,
            0.15861132814448145,
            0.07788788182581427,
            0.2033127058413462,
            0.11406283238286104,
            0.020870012361419343,
            0.2719578430552649,
            0.15170700765742606,
            0.07022129511933332,
            0.04843952826362164,
            0.2555544578976192,
            0.06681340738035137,
            0.15586613310716685,
            0.018646851863471762,
            0.1979297092698637,
            0.23753411269988997,
            0.21343310337377225,
            0.08016910818336964,
            0.1464783972931314,
            0.025950651929534232,
            0.1449831966165307,
            0.23373068876892722,
            0.11880598168141776,
            0.19121899600576953,
            0.15179084381364852,
            0.04935589775227154,
            0.1560593153225572,
            0.04411343229804097,
            0.2392666728301238,
            0.1484349911958273,
            0.17060958026859915,
            0.01433037040599356,
        ],
        [
            0.16353440643700792,
            0.10129020125571776,
            0.06907539765598648,
            0.10707750259980159,
            0.0910430491609086,
            0.09191184484141025,
            0.13076652451317197,
            0.14870505851042468,
            0.06069752009721213,
            0.1047252743607885,
            0.05142652727905344,
            0.13423996349664308,
            0.07531152759015192,
            0.013779707893699597,
            0.17956384365300332,
            0.10016660338980309,
            0.04636455972831437,
            0.03198285359980099,
            0.16873328677427074,
            0.0441144557626597,
            0.10291272221322205,
            0.012311836110395775,
            0.1306857672142806,
            0.1568351101623881,
            0.14092209282890691,
            0.05293273783140711,
            0.09671434268855209,
            0.017134268875714943,
            0.09572711622173373,
            0.1543238480770115,
            0.07844325605769907,
            0.12625492803046978,
            0.10022195734569302,
            0.03258789894704945,
            0.10304027338340047,
            0.029126490235301297,
            0.15797905641835802,
            0.09800621027247283,
            0.11264728258206483,
            0.009461820854890907,
            0.07619899497188476,
            0.06435512726649247,
            0.14418724370624061,
            0.020482591380646117,
            0.18622326856315144,
            0.17680302406232687,
            0.11411324587011096,
            0.04559115895133764,
            0.024318253167761508,
            0.0777066933426362,
            0.16696701026147626,
            0.06641466684484068,
            0.06421006278332052,
            0.1069248857665583,
            0.12159103864374035,
            0.05073753945931583,
            0.04924168633871106,
            0.12722583436268306,
            0.18563179386637813,
            0.0687128870870491,
            0.08905886988994213,
            0.009035432164352614,
            0.049561446318667116,
            0.1347085017272271,
            0.1200359392924501,
            0.027268283817115106,
            0.04224558795450571,
            0.04315262171954351,
            0.05403346926811662,
            0.0023262859537098515,
            0.010721414504714139,
            0.13403654903491943,
            0.10987330337776338,
            0.013298908898025385,
            0.14340914694684173,
            0.17825169707180502,
            0.10868078009526573,
            0.0016445884181541684,
            0.07063958523304881,
            0.1101738445443299,
            0.11758054831650003,
            0.060827727766064016,
            0.14865227744166581,
            0.11702503201869387,
            0.09342064187053477,
            0.15738531812684184,
            0.03201521439082036,
            0.06114448967602477,
            0.10382777552410098,
            0.1431354723068435,
            0.18877393058117947,
            0.04238816261102419,
        ],
        [
            0.1490765359911958,
            0.09233526241995971,
            0.06296852894216698,
            0.0976109157574458,
            0.08299404810701023,
            0.08378603465806785,
            0.11920562114578914,
            0.13555823199591513,
            0.0553313289630816,
            0.09546664504785775,
            0.04687997291732666,
            0.12237197777320692,
            0.06865333050063585,
            0.012561461312757724,
            0.16368883089666822,
            0.09131100042306806,
            0.042265527528094815,
            0.029155289884568912,
            0.153815789880318,
            0.040214352413767675,
            0.09381433834769744,
            0.011223362220940966,
            0.11913200349775391,
            0.1429695160437834,
            0.12846334848596738,
            0.04825302129601608,
            0.08816394973267236,
            0.015619449792770343,
            0.08726400271162271,
            0.1406802714694426,
            0.07150818680750592,
            0.11509288921319505,
            0.0913614606055903,
            0.029706843936407743,
            0.0939306128799265,
            0.026551453999575543,
            0.14401232745525105,
            0.08934160493420577,
            0.10268827852213556,
            0.008625313216639324,
            0.06946233801138785,
            0.05866557169947423,
            0.13143983149579208,
            0.018671751331583056,
            0.16975950445661273,
            0.161172091881058,
            0.1040246378463513,
            0.04156050213755719,
            0.022168307101803907,
            0.07083674267232855,
            0.15220566764447255,
            0.06054302998342982,
            0.058533332184146566,
            0.09747179158578337,
            0.11084132839998145,
            0.046251897641031985,
            0.044888291006624594,
            0.11597795894213936,
            0.1692203212911706,
            0.06263806747503713,
            0.08118528762078707,
            0.008236621012006807,
            0.04517978141038145,
            0.12279909313025451,
            0.10942371341935776,
            0.0248575292652322,
            0.038510708849485854,
            0.0393375529042932,
            0.049256438455844605,
            0.002120621948981863,
            0.00977354778184982,
            0.12218654692727692,
            0.10015954331772974,
            0.012123169149384955,
            0.13073052528871054,
            0.16249268953840867,
            0.09907244951333158,
            0.0014991924320470194,
            0.06439442866999417,
            0.10043351401912919,
            0.1071854004601844,
            0.055450025136289674,
            0.13551011723482043,
            0.10667899665704653,
            0.08516144084638168,
            0.14347108081661994,
            0.029184789698902848,
            0.0557387825256351,
            0.026753030352160315,
            0.057506912662784994,
            0.16716809419341475,
            0.11576080110723733,
            0.01392399094397694,
            0.0701719259281324,
            0.06969711242341488,
            0.1702065083755876,
            0.13916687524684004,
            0.05350498510320062,
            0.1415754204215561,
            0.06957602019036072,
            0.15092969351716987,
            0.028763669977259567,
            0.09378263985254962,
            0.07190282975849477,
            0.09464849295042106,
            0.13048104587817877,
            0.1720846656652933,
            0.03864067866059161,
        ],
    ]

    # Xs[i] is coordinates for a tangent vector at ps[i]
    Xs = [
        [-0.5, -0.4, 0.0, 0.2, -0.5, 0.5, -0.3, 0.9, -0.8, 0.6],
        [0.2, -0.3, 0.2, 0.7, 0.5, 0.0, 0.1, 0.0],
        [-0.3, -0.2, -0.5, 0.6, 0.1, 0.3, 0.0, zeros(16)...],
        [-0.3, 0.9, 0.3, 0.9, 0.1, zeros(16)...],
        [-0.3, 0.5, 0.9, 0.3, -0.3, -0.3, 0.4, -0.8, -0.3, 0.5],
        [0.3, -0.4, 0.3, 0.8, -0.5, -0.5, -0.4, 0.2, -0.7, -0.8],
        [0.5, 0.9, -0.8, 0.3, -0.7, 0.1, -0.1, zeros(16)...],
        [-0.6, 0.0, -0.1, 0.1, -0.7, -0.1, 0.2, -0.0, 0.8, -0.5, 0.9, zeros(16)...],
    ]

    for (M, V, p, q, v, u, dy, X) in zip(Ms, Vs, ps, qs, vs, us, dys, Xs)
        @testset "Manifold $M" begin
            @testset "is_point" begin
                @test is_point(M, p)
                @test is_point(M, q)
                @test_throws DomainError is_point(
                    M,
                    [[1.0, 0.0], p[2:end]...];
                    error=:error,
                )
                @test_throws DomainError is_point(M, [[-1.0], p[2:end]...]; error=:error)
                @test_throws DomainError is_point(M, [p[1], 2 * p[2:end]...]; error=:error)
            end

            @testset "is_vector" begin
                @test is_vector(M, p, v; error=:error)
                @test is_vector(M, p, u; error=:error)
                @test_throws DomainError is_vector(
                    M,
                    [[1.0, 0.0], p[2:end]...],
                    v,
                    false,
                    true,
                )
                @test_throws DomainError is_vector(
                    M,
                    p,
                    [[1.0, 0.0], v[2:end]...],
                    false,
                    true,
                )
                @test_throws DomainError is_vector(M, p, p, false, true)
            end

            Random.seed!(1)
            @testset "rand" begin
                @test is_point(M, rand(M))
                @test is_vector(M, p, rand(M, vector_at=p))
            end

            @testset "get_embedding" begin
                @test get_embedding(M) == Euclidean(prod(V))
            end

            @testset "embed!" begin
                # points
                p_ = zeros(prod(V))
                p__ = zeros(prod(V))
                embed!(M, p_, p)
                embed!(M, p__, [p[1], [-x for x in p[2:end]]...])
                @test is_point(get_embedding(M), p_)
                @test isapprox(p_, (-1)^length(V) * p__)

                # vectors
                v_ = zeros(prod(V))
                embed!(M, v_, p, v)
                @test is_vector(get_embedding(M), p_, v_)
            end

            @testset "get_coordinates" begin
                @test isapprox(v, get_vector(M, p, get_coordinates(M, p, v)))
                @test isapprox(X, get_coordinates(M, p, get_vector(M, p, X)))
                @test isapprox(
                    dot(X, get_coordinates(M, p, v)),
                    inner(M, p, v, get_vector(M, p, X)),
                )
            end

            @testset "exp" begin
                # Zero vector
                p_ = exp(M, p, zeros.(size.(v)))
                @test is_point(M, p_)
                @test isapprox(p, p_; atol=1e-5)

                # Tangent vector in the scaling direction
                p_ = exp(M, p, [v[1], zeros.(size.(v[2:end]))...])
                @test is_point(M, p_)
                @test isapprox([p[1] + v[1], p[2:end]...], p_; atol=1e-5)

                # Generic tangent vector
                p_ = exp(M, p, v)
                @test is_point(M, p)

                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * v)), -1.0)
                @test isapprox(geodesic_speed, -norm(M, p, v); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * v)), -0.811)
                @test isapprox(geodesic_speed, -norm(M, p, v); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * v)), -0.479)
                @test isapprox(geodesic_speed, -norm(M, p, v); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * v)), 0.181)
                @test isapprox(geodesic_speed, norm(M, p, v); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * v)), 0.703)
                @test isapprox(geodesic_speed, norm(M, p, v); atol=1e-5)
                geodesic_speed =
                    central_fdm(3, 1)(t -> distance(M, p, exp(M, p, t * v)), 1.0)
                @test isapprox(geodesic_speed, norm(M, p, v); atol=1e-5)

                # Geodesics are (locally) length-minizing. So let B_a be a one-parameter
                # family of curves such that B_0 is a geodesic. Then the derivative of
                # length(B_a) at a = 0 should be 0, and the second derivative should be
                # nonnegative.

                n = manifold_dimension(M)
                x = get_coordinates(M, p, v)
                x0 = 0.0 * x
                x1 = 0.2 * x
                x2 = 0.4 * x
                x3 = 0.6 * x
                x4 = 0.8 * x
                x5 = 1.0 * x

                function curve_length(y::Vector{Float64})
                    @assert(length(y) == 4 * n)

                    # Control points
                    y1 = y[1:n]
                    y2 = y[(n + 1):(2 * n)]
                    y3 = y[(2 * n + 1):(3 * n)]
                    y4 = y[(3 * n + 1):(4 * n)]

                    # Bezier curve from 0 to v with control points y1, ..., y4
                    function b(t)
                        return (
                            (1 - t)^5 * x0 +
                            5 * t * (1 - t)^4 * (x1 + y1) +
                            10 * t^2 * (1 - t)^3 * (x2 + y2) +
                            10 * t^3 * (1 - t)^2 * (x3 + y3) +
                            5 * t^4 * (1 - t) * (x4 + y4) +
                            t^5 * x5
                        )
                    end

                    # Length of curve on manifold
                    ps_ = [exp(M, p, get_vector(M, p, b(t))) for t in 0.0:1e-3:1.0]
                    ds = [
                        distance(M, p1, p2) for
                        (p1, p2) in zip(ps_[1:(end - 1)], ps_[2:end])
                    ]
                    return sum(ds)
                end

                # dy = rand(4 * n); dy = dy / norm(dy)
                f = a -> curve_length(a * dy)
                @test isapprox(central_fdm(3, 1)(f, 0.0), 0.0; atol=1e-5)
                @test central_fdm(3, 2)(f, 0.0) >= 0.0
            end

            @testset "log" begin
                # Same point
                v_ = log(M, p, p)
                @test is_vector(M, p, v_)
                @test isapprox(zeros.(size.(v)), v_; atol=1e-5)

                # Scaled point
                v_ = log(M, p, [q[1], p[2:end]...])
                @test is_vector(M, p, v_)
                @test isapprox(v_, [q[1] - p[1], zeros.(size.(q[2:end]))...]; atol=1e-5)

                # Generic tangent vector
                v_ = log(M, p, q)
                @test is_vector(M, p, v_)
            end

            @testset "norm" begin
                @test isapprox(norm(M, p, log(M, p, q)), distance(M, p, q))
            end

            @testset "sectional_curvature" begin
                # Test that sectional curvature is difference between circumference
                # and 2 pi r for small circles.

                # Orthonormalize
                u_ = u / norm(M, p, u)
                v_ = v - inner(M, p, u_, v) * u_
                v_ = v_ / norm(M, p, v_)

                r = 1e-2
                ps_ = [
                    exp(M, p, r * (cos(theta) * u_ + sin(theta) * v_)) for
                    theta in 0.0:1e-3:(2 * pi)
                ]
                ds = [distance(M, p1, p2) for (p1, p2) in zip(ps_, [ps_[2:end]..., ps_[1]])]
                C = sum(ds)
                K = 3 * (2 * pi * r - C) / (pi * r^3) # https://en.wikipedia.org/wiki/Bertrand%E2%80%93Diguet%E2%80%93Puiseux_theorem

                @test isapprox(K, sectional_curvature(M, p, u, v); rtol=1e-2, atol=1e-2)
            end

            @testset "riemann_tensor" begin
                @test isapprox(
                    sectional_curvature(M, p, u, v),
                    inner(M, p, riemann_tensor(M, p, u, v, v), u) /
                    (inner(M, p, u, u) * inner(M, p, v, v) - inner(M, p, u, v)^2),
                )
            end
        end
    end

    # Test a point that does not use the closest representative
    @testset "log" begin
        M = Ms[4]
        p = ps[4]
        q = qs[4]
        q_ = [q[1], q[2], q[3], q[4], -q[5]]
        @test is_vector(M, p, log(M, p, q_))

        M = Ms[8]
        p = ps[8]
        q = qs[8]
        q_ = [q[1], q[2], q[3], q[4], -q[5]]
        @test is_vector(M, p, log(M, p, q_))
    end
end
