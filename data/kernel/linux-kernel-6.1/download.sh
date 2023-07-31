if [ -f "bzImage" ] && \
        [ -f "snapshot.img" ] && \
        [ -f "vmlinux.map" ] && \
        [ -f "vmlinux.dis" ] && \
        [ -f "block-info" ] && \
        [ -f "block-calling" ] ; then
        echo "Linux kernel-6.1 data is already downloaded"
        exit 0
fi

wget -O bzImage "https://www.dropbox.com/scl/fi/4eb2tjlff9563wfphg0sh/bzImage?rlkey=vzkuicrc0skw70bspdxhe1tz0&dl=0"
wget -O snapshot.img "https://www.dropbox.com/scl/fi/29ghp23xqt7r1luv29f4b/snapshot.img?rlkey=tthpaxcjcvwy80yzipu5briuz&dl=0"
wget -O vmlinux.dis "https://www.dropbox.com/scl/fi/w8f5gmayqksrvyx7lfxu9/vmlinux.dis?rlkey=jx8btau9b6fuwfoagqqoy4x01&dl=0"
wget -O vmlinux.map "https://www.dropbox.com/scl/fi/22tti26frgmj96c9iokl3/vmlinux.map?rlkey=h7aoqnyxsse12qlt9bvx9wwnp&dl=0"
wget -O block-calling "https://www.dropbox.com/scl/fi/o15co59qadhk1bievyrvd/block-calling-dict?rlkey=xmhcefisa6ozfbv4ejktko9q0&dl=0"
wget -O block-info "https://www.dropbox.com/scl/fi/2lija3dqiya40pj8ubx8w/block-info?rlkey=dk1vjtlc6a5ro1cke1unm4og8&dl=0"
