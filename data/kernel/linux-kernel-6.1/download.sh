if [ -f "bzImage" ] && \
        [ -f "snapshot.img" ] && \
        [ -f "vmlinux.map" ] && \
        [ -f "vmlinux.dis" ] && \
        [ -f "block-info" ] && \
        [ -f "block-calling" ] ; then
        echo "Linux kernel-6.1 data is already downloaded"
        exit 0
fi

wget -O bzImage "https://purdue0-my.sharepoint.com/:u:/g/personal/sishuai_purdue_edu/EcvhzdHPxBdIus8-ewZoxk4BB6YmdWFjobjbLWTkBKkZDA?e=OYCga7&download=1"
wget -O snapshot.img "https://purdue0-my.sharepoint.com/:u:/g/personal/sishuai_purdue_edu/Eb-rFhD8NfRNocbP5-OFBvMBKNN-yoo_1-TyPp1xqxVyfg?e=0cvNEd&download=1"
wget -O vmlinux.dis "https://purdue0-my.sharepoint.com/:u:/g/personal/sishuai_purdue_edu/EdeiQoKYwp5Jqk2hNceKzSsBdGUa-Ha4SWtpnzEkYRpScg?e=YuCD7E&download=1"
wget -O vmlinux.map "https://purdue0-my.sharepoint.com/:u:/g/personal/sishuai_purdue_edu/ETh-MrOxzTFGt3JbTkhQFYwBU2VtF7RRk_jNZ4NlDLLU2w?e=tAn38g&download=1"
wget -O block-calling "https://purdue0-my.sharepoint.com/:u:/g/personal/sishuai_purdue_edu/ESeeruTgw_FOivRVMLAQiDcBA2hFMzXRtqOsAY12WN9PiQ?e=oumbM4&download=1"
wget -O block-info "https://purdue0-my.sharepoint.com/:u:/g/personal/sishuai_purdue_edu/EeNay503G6pBvHszHRF6GeIBuWbXJwqTyxoUKByqCqkYdQ?e=THYBdC&download=1"
